"""Скрипт обучения / дообучения мультимодального BERT-классификатора.

Зачем: дать возможность улучшать модель на собственных размеченных данных.

Использование:
    python -m ml_models.train_bert \\
        --csv data/train.csv \\
        --output models/best_multimodal_bert.pt \\
        --epochs 5 --batch-size 16 --lr 2e-5

Формат CSV (разделитель `,`):
    image_path,ocr_text,yolo_classes,yolo_confs,topic
    creatives/img_001.jpg,"sale 50% off","tie;handbag","0.91;0.74",ties
    creatives/img_002.jpg,"summer cups","cup;bowl","0.88;0.65",cups

- yolo_classes/yolo_confs: значения через `;`. Пустая строка = нет детекций.
- topic: один из TOPIC_LABELS (см. backend/config.py).

Можно стартовать с нуля или дообучать поверх существующего чекпоинта (--resume).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from config import TOPIC_LABELS
from ml_models.classifier import MAX_TOKEN_LENGTH
from ml_models.classifier import MultiModalBertClassifier
from ml_models.clip_extractor import CLIP_FEATURE_DIM
from ml_models.preprocessing import clean_text_for_bert
from ml_models.preprocessing import yolo_to_vector_for_bert


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_bert")

LABEL2IDX = {t: i for i, t in enumerate(TOPIC_LABELS)}


def load_clip_features(npz_path: Path) -> dict[str, np.ndarray]:
    """Загружает предвычисленные CLIP-фичи из .npz → dict[image_path → vector(512)]."""
    data = np.load(npz_path, allow_pickle=True)
    paths = data["paths"]
    features = data["features"]
    if features.shape[1] != CLIP_FEATURE_DIM:
        raise ValueError(
            f"Ожидалась размерность CLIP={CLIP_FEATURE_DIM}, получено {features.shape[1]}",
        )
    logger.info("Загружено %d CLIP-векторов из %s", len(paths), npz_path)
    return {str(p): features[i].astype(np.float32) for i, p in enumerate(paths)}


# ────────────────────────── Dataset ──────────────────────────
class CreativeDataset(Dataset):
    """Датасет (text + clip + yolo-vector + label) для мультимодального BERT."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: BertTokenizer,
        clip_features: dict[str, np.ndarray] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.clip_features = clip_features or {}
        self._zero_clip = np.zeros(CLIP_FEATURE_DIM, dtype=np.float32)
        self._missing_clip_reported = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        text = clean_text_for_bert(row.get("ocr_text") or "")

        classes_raw = str(row.get("yolo_classes") or "")
        confs_raw = str(row.get("yolo_confs") or "")
        classes = [c.strip() for c in classes_raw.split(";") if c.strip()]
        confs = [float(c) for c in confs_raw.split(";") if c.strip()]
        yolo_vec = yolo_to_vector_for_bert(classes, confs)

        image_path = str(row.get("image_path") or "")
        clip_vec = self.clip_features.get(image_path)
        if clip_vec is None:
            clip_vec = self._zero_clip

        topic = str(row["topic"]).strip()
        if topic not in LABEL2IDX:
            raise ValueError(f"Неизвестный топик '{topic}'. Допустимы: {TOPIC_LABELS}")
        label = LABEL2IDX[topic]

        encoded = self.tokenizer(
            text,
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "yolo_vector": torch.tensor(yolo_vec, dtype=torch.float32),
            "clip_vector": torch.tensor(clip_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ────────────────────────── Train / eval ──────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        yolo_vec = batch["yolo_vector"].to(device)
        clip_vec = batch["clip_vector"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, yolo_vec, clip_vec)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        yolo_vec = batch["yolo_vector"].to(device)
        clip_vec = batch["clip_vector"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask, yolo_vec, clip_vec)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(1, len(loader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


# ────────────────────────── Main ──────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Тренировка мультимодального BERT")
    p.add_argument("--csv", required=True, type=Path, help="CSV с тренировочными данными")
    p.add_argument("--output", required=True, type=Path, help="Куда сохранить веса (.pt)")
    p.add_argument("--resume", type=Path, default=None, help="Чекпоинт для дообучения")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--freeze-bert", action="store_true",
                   help="Заморозить BERT, обучать только голову (быстрее, для маленьких датасетов).")
    p.add_argument("--unfreeze-last", type=int, default=0,
                   help="Разморозить последние N слоёв энкодера BERT (0 = ничего, 2-4 рекомендуется).")
    p.add_argument("--clip-features", type=Path, default=None,
                   help="Путь к .npz с предвычисленными CLIP-эмбеддингами изображений.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.csv.exists():
        logger.error("CSV не найден: %s", args.csv)
        return 1

    df = pd.read_csv(args.csv)
    required = {"ocr_text", "yolo_classes", "yolo_confs", "topic"}
    missing = required - set(df.columns)
    if missing:
        logger.error("В CSV отсутствуют колонки: %s", missing)
        return 1

    logger.info("Загружено %d примеров", len(df))
    logger.info("Распределение по топикам:\n%s", df["topic"].value_counts().to_string())

    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    clip_features = None
    if args.clip_features is not None:
        if not args.clip_features.exists():
            logger.error("Файл CLIP-фичей не найден: %s", args.clip_features)
            return 1
        clip_features = load_clip_features(args.clip_features)
        missing = sum(1 for p in df["image_path"] if str(p) not in clip_features)
        if missing:
            logger.warning(
                "В CLIP-фичах отсутствует %d/%d строк — им будет подставлен нулевой вектор.",
                missing, len(df),
            )
    else:
        logger.warning("⚠️  --clip-features не указан: CLIP будет нулевым, обучение деградирует")

    full_ds = CreativeDataset(df, tokenizer, clip_features=clip_features)
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("Train: %d, Val: %d", train_size, val_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MultiModalBertClassifier(num_labels=len(TOPIC_LABELS)).to(device)
    if args.resume and args.resume.exists():
        logger.info("⏯  Загружаю чекпоинт: %s", args.resume)
        state_dict = torch.load(args.resume, map_location=device)
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(
                "Чекпоинт несовместим с текущей архитектурой (%s). "
                "Гружу только BERT-энкодер, остальное инициализируем с нуля.",
                str(e).split("\n")[0],
            )
            bert_only = {
                k.replace("bert.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("bert.")
            }
            missing, unexpected = model.bert.load_state_dict(bert_only, strict=False)
            logger.info(
                "BERT загружен частично: missing=%d, unexpected=%d",
                len(missing), len(unexpected),
            )

    if args.freeze_bert:
        logger.info("❄️  Замораживаю BERT (обучается только голова)")
        for p_ in model.bert.parameters():
            p_.requires_grad = False

    if args.unfreeze_last > 0:
        # Сначала всё замораживаем, потом размораживаем последние N слоёв энкодера + pooler
        logger.info("🔓 Размораживаю последние %d слоёв BERT-энкодера", args.unfreeze_last)
        for p_ in model.bert.parameters():
            p_.requires_grad = False
        encoder_layers = model.bert.encoder.layer  # ModuleList из 12 слоёв
        for layer in encoder_layers[-args.unfreeze_last:]:
            for p_ in layer.parameters():
                p_.requires_grad = True
        if hasattr(model.bert, "pooler") and model.bert.pooler is not None:
            for p_ in model.bert.pooler.parameters():
                p_.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("🔢 Обучаемых параметров: %d", sum(p.numel() for p in trainable))
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %d/%d │ train_loss=%.4f │ val_loss=%.4f │ val_acc=%.4f",
            epoch, args.epochs, train_loss, val_loss, val_acc,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.output)
            logger.info("💾 Сохранён лучший чекпоинт (acc=%.4f) → %s", val_acc, args.output)

    logger.info("✅ Обучение завершено. Best val_acc=%.4f", best_acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
