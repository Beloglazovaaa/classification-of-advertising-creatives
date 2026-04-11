"""Обогащение датасета: проход по всем картинкам, OCR + YOLO детекция → CSV.

Заполняет колонки `ocr_text`, `yolo_classes`, `yolo_confs` в train.csv,
чтобы train_bert.py мог обучаться на полноценном мультимодальном входе.

Использование:
    pip install easyocr ultralytics pandas pillow tqdm

    # Запустить из корня проекта (нужен .venv с моделями)
    python scripts/enrich_dataset.py \\
        --csv dataset/train.csv \\
        --output dataset/train_enriched.csv \\
        --models-dir minio_init/models

Если модели не найдены — соответствующие колонки останутся пустыми
(модель сможет обучиться без них, просто хуже).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("enrich")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_LANGS = ["en", "ru"]
YOLO_CONF_THRESHOLD = 0.35
MAX_DETECTIONS = 5


def init_ocr(models_dir: Path):
    """Инициализирует EasyOCR. Возвращает reader или None."""
    try:
        import easyocr
    except ImportError:
        logger.warning("easyocr не установлен → OCR пропускается")
        return None

    weights_dir = models_dir / "easy_ocr"
    if not weights_dir.exists():
        logger.warning("Веса EasyOCR не найдены: %s → OCR пропускается", weights_dir)
        return None

    try:
        return easyocr.Reader(
            OCR_LANGS,
            gpu=False,
            model_storage_directory=str(weights_dir),
            download_enabled=False,
        )
    except Exception as e:
        logger.warning("Не удалось инициализировать EasyOCR: %s", e)
        return None


def init_yolo(models_dir: Path):
    """Инициализирует YOLOv8. Возвращает model или None."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics не установлен → YOLO пропускается")
        return None

    weights_path = models_dir / "yolov8m.pt"
    if not weights_path.exists():
        logger.warning("Веса YOLO не найдены: %s → YOLO пропускается", weights_path)
        return None

    try:
        return YOLO(str(weights_path))
    except Exception as e:
        logger.warning("Не удалось загрузить YOLO: %s", e)
        return None


def run_ocr(reader, image_path: Path) -> str:
    """Возвращает извлечённый текст одной строкой."""
    if reader is None:
        return ""
    try:
        results = reader.readtext(str(image_path))
        texts = [t for (_, t, _) in results if t]
        return " ".join(texts).strip()
    except Exception as e:
        logger.debug("OCR fail %s: %s", image_path.name, e)
        return ""


def run_yolo(model, image_path: Path) -> tuple[str, str]:
    """Возвращает (yolo_classes, yolo_confs) — строки через ;"""
    if model is None:
        return "", ""
    try:
        img = Image.open(image_path).convert("RGB")
        results = model.predict(img, conf=YOLO_CONF_THRESHOLD, verbose=False)

        items: list[tuple[str, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])
                items.append((cls_name, conf))

        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:MAX_DETECTIONS]

        classes = ";".join(c for c, _ in items)
        confs = ";".join(f"{conf:.4f}" for _, conf in items)
        return classes, confs
    except Exception as e:
        logger.debug("YOLO fail %s: %s", image_path.name, e)
        return "", ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Обогащение CSV датасета OCR + YOLO")
    p.add_argument("--csv", type=Path, default=PROJECT_ROOT / "dataset" / "train.csv",
                   help="Входной CSV (от build_dataset.py)")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "dataset" / "train_enriched.csv",
                   help="Куда сохранить обогащённый CSV")
    p.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "minio_init" / "models",
                   help="Путь к директории с весами")
    p.add_argument("--limit", type=int, default=None,
                   help="Обработать только первые N строк (для теста)")
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--skip-yolo", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        logger.error("CSV не найден: %s", args.csv)
        return 1

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    logger.info("Загружено %d строк из %s", len(df), args.csv)

    ocr_reader = None if args.skip_ocr else init_ocr(args.models_dir)
    yolo_model = None if args.skip_yolo else init_yolo(args.models_dir)

    if ocr_reader:
        logger.info("✓ OCR инициализирован (EasyOCR ru+en)")
    if yolo_model:
        logger.info("✓ YOLO инициализирован (YOLOv8m)")

    ocr_texts = []
    yolo_classes_list = []
    yolo_confs_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обогащение"):
        img_rel = row["image_path"]
        img_path = PROJECT_ROOT / img_rel
        if not img_path.exists():
            logger.warning("Файл не найден: %s", img_path)
            ocr_texts.append("")
            yolo_classes_list.append("")
            yolo_confs_list.append("")
            continue

        ocr_texts.append(run_ocr(ocr_reader, img_path))
        cls, confs = run_yolo(yolo_model, img_path)
        yolo_classes_list.append(cls)
        yolo_confs_list.append(confs)

    df["ocr_text"] = ocr_texts
    df["yolo_classes"] = yolo_classes_list
    df["yolo_confs"] = yolo_confs_list

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("💾 Сохранено: %s", args.output)

    # Статистика
    n_with_ocr = (df["ocr_text"].str.len() > 0).sum()
    n_with_yolo = (df["yolo_classes"].str.len() > 0).sum()
    logger.info(
        "Статистика: OCR=%d/%d (%.0f%%), YOLO=%d/%d (%.0f%%)",
        n_with_ocr, len(df), n_with_ocr / len(df) * 100,
        n_with_yolo, len(df), n_with_yolo / len(df) * 100,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
