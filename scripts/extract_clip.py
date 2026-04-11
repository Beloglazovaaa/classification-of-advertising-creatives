"""Предвычисление CLIP-эмбеддингов для всего датасета.

Читает CSV с колонкой image_path, прогоняет CLIP по каждой картинке,
сохраняет результат в single .npz файл с ключами:
    - paths:    np.array(N,) dtype=object  — относительные пути из CSV
    - features: np.array(N, 512) dtype=float32 — L2-нормированные векторы

Пример:
    python scripts/extract_clip.py \\
        --csv dataset/train_enriched.csv \\
        --output dataset/clip_features.npz \\
        --dataset-root /app
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Добавляем backend в path, чтобы импортировать clip_extractor
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from ml_models.clip_extractor import (  # noqa: E402
    CLIP_FEATURE_DIM,
    extract_clip_features,
    zeros_vector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("extract_clip")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute CLIP image features")
    p.add_argument("--csv", required=True, type=Path, help="CSV с колонкой image_path")
    p.add_argument("--output", required=True, type=Path, help="Куда сохранить .npz")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Корень, относительно которого в CSV указан image_path",
    )
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.csv.exists():
        logger.error("CSV не найден: %s", args.csv)
        return 1

    df = pd.read_csv(args.csv)
    if "image_path" not in df.columns:
        logger.error("В CSV нет колонки 'image_path'")
        return 1

    n = len(df)
    logger.info("Обрабатываю %d изображений...", n)

    paths_out: list[str] = []
    features_out = np.zeros((n, CLIP_FEATURE_DIM), dtype=np.float32)

    ok, failed = 0, 0
    t0 = time.time()

    for i, row in df.iterrows():
        rel_path = str(row["image_path"])
        abs_path = args.dataset_root / rel_path
        if not abs_path.exists():
            # пробуем как есть (возможно, уже абсолютный)
            abs_path = Path(rel_path)

        feats = extract_clip_features(abs_path) if abs_path.exists() else None
        if feats is None:
            features_out[i] = zeros_vector()
            failed += 1
        else:
            features_out[i] = feats
            ok += 1

        paths_out.append(rel_path)

        if (i + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            logger.info(
                "%d/%d │ ok=%d failed=%d │ %.1f img/s │ ETA %.0fs",
                i + 1, n, ok, failed, rate, eta,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        paths=np.array(paths_out, dtype=object),
        features=features_out,
    )

    elapsed = time.time() - t0
    logger.info(
        "✅ Готово: %d ok, %d failed, %.0fs. Сохранено → %s (%.1f MB)",
        ok, failed, elapsed, args.output,
        args.output.stat().st_size / 1024 / 1024,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
