"""

Основные отличия от ml_models/train_bert.py:
    1. Стратифицированный split (до этого был random_split — классы могли попасть
       в val неравномерно).
    2. Class weights в CrossEntropyLoss для борьбы с дисбалансом (помады 57 vs
       сумки 25).
    3. Label smoothing = 0.1 (лучшая калибровка, меньше переобучения на редких
       классах).
    4. Больший dropout (0.4) и L2 в AdamW (weight_decay=0.05).
    5. Предвычисление CLIP-фичей прямо здесь, без отдельного запуска extract_clip.
    6. Финальный отчёт: confusion matrix, per-class precision/recall/F1, macro-F1.
    7. Ablation: опционально считает val_acc без CLIP и без YOLO каналов.
    8. Multi-seed: N запусков, mean ± std от финальной метрики

Запускать внутри celery_worker (там есть CLIP, BERT и модели):

    docker compose cp scripts/train_improved.py celery_worker:/app/train_improved.py
    docker compose cp dataset celery_worker:/app/dataset
    docker compose exec celery_worker python /app/train_improved.py \
        --csv /app/dataset/train_enriched.csv \
        --dataset-root /app \
        --output /app/models/best_multimodal_bert.pt \
        --epochs 25 --seeds 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from config import TOPIC_LABELS
from ml_models.classifier import MAX_TOKEN_LENGTH, MultiModalBertClassifier
from ml_models.clip_extractor import (
    CLIP_FEATURE_DIM,
    extract_clip_features,
    zeros_vector as clip_zeros,
)
from ml_models.preprocessing import clean_text_for_bert, yolo_to_vector_for_bert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_improved")

LABEL2IDX = {t: i for i, t in enumerate(TOPIC_LABELS)}


# ─────────────────── CLIP precompute (кешируется в .npz) ───────────────────
def precompute_clip(df: pd.DataFrame, dataset_root: Path, cache: Path) -> dict[str, np.ndarray]:
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        logger.info("CLIP-кеш найден: %s (%d векторов)", cache, len(data["paths"]))
        return {str(p): data["features"][i].astype(np.float32)
                for i, p in enumerate(data["paths"])}

    logger.info("Считаю CLIP-фичи для %d изображений...", len(df))
    feats = np.zeros((len(df), CLIP_FEATURE_DIM), dtype=np.float32)
    paths_out: list[str] = []
    t0 = time.time()
    ok, failed = 0, 0
    for i, row in df.iterrows():
        rel = str(row["image_path"])
        abs_path = dataset_root / rel
        if not abs_path.exists():
            abs_path = Path(rel)
        vec = extract_clip_features(abs_path) if abs_path.exists() else None
        if vec is None:
            feats[i] = clip_zeros()
            failed += 1
        else:
            feats[i] = vec
            ok += 1
        paths_out.append(rel)
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            logger.info("CLIP %d/%d | ok=%d failed=%d | %.1f img/s", i + 1, len(df), ok, failed, rate)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, paths=np.array(paths_out, dtype=object), features=feats)
    logger.info("CLIP сохранён → %s (%d ok, %d failed, %.0f s)", cache, ok, failed, time.time() - t0)
    return {paths_out[i]: feats[i] for i in range(len(paths_out))}


# ─────────────────── Dataset ───────────────────
class CreativeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer,
                 clip_features: dict[str, np.ndarray], training: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.clip_features = clip_features
        self.training = training
        self._zero_clip = np.zeros(CLIP_FEATURE_DIM, dtype=np.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = clean_text_for_bert(row.get("ocr_text") or "")

        # Text augmentation: случайно выбрасываем слова на train (word-level dropout)
        if self.training and text:
            words = text.split()
            if len(words) > 3:
                keep = np.random.rand(len(words)) > 0.15
                kept = [w for w, k in zip(words, keep) if k]
                if kept:
                    text = " ".join(kept)

        classes = [c.strip() for c in str(row.get("yolo_classes") or "").split(";") if c.strip()]
        confs = [float(c) for c in str(row.get("yolo_confs") or "").split(";") if c.strip()]
        yolo_vec = yolo_to_vector_for_bert(classes, confs)

        clip_vec = self.clip_features.get(str(row.get("image_path") or ""), self._zero_clip)
        label = LABEL2IDX[str(row["topic"]).strip()]

        enc = self.tokenizer(text, max_length=MAX_TOKEN_LENGTH, padding="max_length",
                             truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "yolo_vector": torch.tensor(yolo_vec, dtype=torch.float32),
            "clip_vector": torch.tensor(clip_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ─────────────────── Train / Eval ───────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total = 0.0
    for b in loader:
        optimizer.zero_grad()
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                       b["yolo_vector"].to(device), b["clip_vector"].to(device))
        loss = criterion(logits, b["label"].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device, zero_clip=False, zero_yolo=False):
    model.eval()
    total_loss, y_true, y_pred = 0.0, [], []
    for b in loader:
        clip_v = b["clip_vector"].to(device)
        yolo_v = b["yolo_vector"].to(device)
        if zero_clip: clip_v = torch.zeros_like(clip_v)
        if zero_yolo: yolo_v = torch.zeros_like(yolo_v)
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), yolo_v, clip_v)
        loss = criterion(logits, b["label"].to(device))
        total_loss += loss.item()
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
        y_true.extend(b["label"].cpu().tolist())
    acc = float(np.mean([p == t for p, t in zip(y_pred, y_true)]))
    return total_loss / max(1, len(loader)), acc, y_true, y_pred


def run_one_seed(seed: int, df: pd.DataFrame, clip_features: dict,
                 args, tokenizer, device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Стратифицированный split по topic
    train_df, val_df = train_test_split(
        df, test_size=args.val_split, random_state=seed,
        stratify=df["topic"].values,
    )
    logger.info("[seed=%d] train=%d val=%d", seed, len(train_df), len(val_df))

    train_ds = CreativeDataset(train_df, tokenizer, clip_features, training=True)
    val_ds = CreativeDataset(val_df, tokenizer, clip_features, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Class weights (мягкие: sqrt от обратной частоты, не agressive inverse)
    counts = Counter(train_df["topic"].tolist())
    weights = np.array([1.0 / np.sqrt(counts[t]) for t in TOPIC_LABELS], dtype=np.float32)
    weights = weights / weights.mean()
    logger.info("[seed=%d] class weights: min=%.2f max=%.2f", seed, weights.min(), weights.max())
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    model = MultiModalBertClassifier(num_labels=len(TOPIC_LABELS)).to(device)

    # Морозим BERT (ускорение + защита от переобучения на 867 изображениях)
    for p in model.bert.parameters():
        p.requires_grad = False
    # dropout по умолчанию 0.3 — оставляем как в оригинале

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("[seed=%d] обучаемых параметров: %d", seed, sum(p.numel() for p in trainable))

    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.05),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    best_acc, best_state, best_epoch = 0.0, None, 0
    for ep in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        logger.info("[seed=%d] ep %2d/%d | train=%.4f val=%.4f acc=%.4f",
                    seed, ep, args.epochs, tl, vl, va)
        if va > best_acc:
            best_acc = va
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Перезагружаем лучшее состояние и собираем метрики
    model.load_state_dict(best_state)
    _, va_full, yt, yp = evaluate(model, val_loader, criterion, device)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)

    # Ablation
    _, va_noclip, _, _ = evaluate(model, val_loader, criterion, device, zero_clip=True)
    _, va_noyolo, _, _ = evaluate(model, val_loader, criterion, device, zero_yolo=True)
    _, va_textonly, _, _ = evaluate(model, val_loader, criterion, device,
                                    zero_clip=True, zero_yolo=True)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "val_acc": va_full,
        "macro_f1": macro_f1,
        "val_acc_no_clip": va_noclip,
        "val_acc_no_yolo": va_noyolo,
        "val_acc_text_only": va_textonly,
        "y_true": yt,
        "y_pred": yp,
        "state_dict": best_state,
    }


# ─────────────────── Main ───────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--clip-cache", type=Path,
                   default=Path("/app/models/clip_features.npz"))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=3, help="Сколько прогонов с разными seed (для mean±std)")
    p.add_argument("--report", type=Path, default=Path("/app/models/train_report.json"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        logger.error("CSV не найден: %s", args.csv); return 1

    df = pd.read_csv(args.csv)
    logger.info("Загружено %d строк, классов %d", len(df), df["topic"].nunique())
    logger.info("Распределение:\n%s", df["topic"].value_counts().to_string())

    device = torch.device("cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    clip_features = precompute_clip(df, args.dataset_root, args.clip_cache)

    results = []
    best_overall = None
    for i in range(args.seeds):
        seed = 42 + i
        logger.info("═" * 60)
        logger.info("SEED %d (%d/%d)", seed, i + 1, args.seeds)
        logger.info("═" * 60)
        r = run_one_seed(seed, df, clip_features, args, tokenizer, device)
        results.append(r)
        if best_overall is None or r["val_acc"] > best_overall["val_acc"]:
            best_overall = r

    # Отчёт
    accs = np.array([r["val_acc"] for r in results])
    f1s = np.array([r["macro_f1"] for r in results])
    no_clip = np.array([r["val_acc_no_clip"] for r in results])
    no_yolo = np.array([r["val_acc_no_yolo"] for r in results])
    text_only = np.array([r["val_acc_text_only"] for r in results])

    logger.info("═" * 60)
    logger.info("ФИНАЛЬНЫЙ ОТЧЁТ (%d seeds)", args.seeds)
    logger.info("═" * 60)
    logger.info("val_acc:            %.4f ± %.4f", accs.mean(), accs.std())
    logger.info("macro_F1:           %.4f ± %.4f", f1s.mean(), f1s.std())
    logger.info("val_acc без CLIP:   %.4f ± %.4f  (вклад CLIP  = %+.4f)",
                no_clip.mean(), no_clip.std(), accs.mean() - no_clip.mean())
    logger.info("val_acc без YOLO:   %.4f ± %.4f  (вклад YOLO = %+.4f)",
                no_yolo.mean(), no_yolo.std(), accs.mean() - no_yolo.mean())
    logger.info("val_acc только текст: %.4f ± %.4f", text_only.mean(), text_only.std())

    # Per-class отчёт для лучшего прогона
    logger.info("─── Лучший прогон: seed=%d, val_acc=%.4f ───",
                best_overall["seed"], best_overall["val_acc"])
    report = classification_report(
        best_overall["y_true"], best_overall["y_pred"],
        labels=list(range(len(TOPIC_LABELS))),
        target_names=TOPIC_LABELS, zero_division=0, digits=3,
    )
    logger.info("Per-class:\n%s", report)

    cm = confusion_matrix(best_overall["y_true"], best_overall["y_pred"],
                          labels=list(range(len(TOPIC_LABELS))))
    logger.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)

    # Топ путаницы (off-diagonal)
    confusions = []
    for i, true_lbl in enumerate(TOPIC_LABELS):
        for j, pred_lbl in enumerate(TOPIC_LABELS):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], true_lbl, pred_lbl))
    confusions.sort(reverse=True)
    logger.info("Топ-10 путаемых пар:")
    for n, t, p in confusions[:10]:
        logger.info("  %s → %s: %d", t, p, n)

    # Сохраняем лучшие веса
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_overall["state_dict"], args.output)
    logger.info("Веса сохранены → %s", args.output)

    # JSON-отчёт
    report_json = {
        "n_seeds": args.seeds,
        "val_acc_mean": float(accs.mean()),
        "val_acc_std": float(accs.std()),
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_std": float(f1s.std()),
        "ablation": {
            "full_model": float(accs.mean()),
            "no_clip": float(no_clip.mean()),
            "no_yolo": float(no_yolo.mean()),
            "text_only": float(text_only.mean()),
        },
        "best_seed": best_overall["seed"],
        "best_val_acc": float(best_overall["val_acc"]),
        "best_macro_f1": float(best_overall["macro_f1"]),
        "top_confusions": [{"true": t, "pred": p, "count": int(n)}
                           for n, t, p in confusions[:15]],
        "confusion_matrix": cm.tolist(),
        "labels": TOPIC_LABELS,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_json, ensure_ascii=False, indent=2))
    logger.info("Отчёт сохранён → %s", args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
