"""CLIP image encoder: извлекает 512-мерный вектор изображения.

Используется:
- при обучении (scripts/extract_clip.py предвычисляет фичи в npz)
- на инференсе (classify_creative считает фичи on-the-fly)

Ленивая загрузка singleton. Если CLIP недоступен — возвращает None/zeros,
чтобы классификатор мог работать без визуальной модальности (деградация, не падение).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_FEATURE_DIM = 512

_clip_model = None
_clip_processor = None
_clip_unavailable = False


def get_clip():
    """Возвращает (model, processor) или (None, None), если загрузка не удалась."""
    global _clip_model, _clip_processor, _clip_unavailable

    if _clip_unavailable:
        return None, None
    if _clip_model is not None and _clip_processor is not None:
        return _clip_model, _clip_processor

    try:
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Загружаю CLIP (%s)...", CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
        logger.info("CLIP загружен (feature_dim=%d)", CLIP_FEATURE_DIM)
    except Exception as e:
        logger.warning("CLIP недоступен (%s). Изображения будут заменены нулями.", e)
        _clip_unavailable = True
        _clip_model = None
        _clip_processor = None
        return None, None

    return _clip_model, _clip_processor


@torch.no_grad()
def extract_clip_features(image_path: str | Path) -> Optional[np.ndarray]:
    """Возвращает L2-нормированный 512-мерный вектор или None при ошибке."""
    model, processor = get_clip()
    if model is None or processor is None:
        return None

    try:
        img = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        logger.warning("Не удалось открыть изображение %s: %s", image_path, e)
        return None

    try:
        inputs = processor(images=img, return_tensors="pt")
        features = model.get_image_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.warning("Ошибка CLIP-инференса для %s: %s", image_path, e)
        return None


def zeros_vector() -> np.ndarray:
    """Возвращает нулевой вектор нужной размерности (fallback, если CLIP упал)."""
    return np.zeros(CLIP_FEATURE_DIM, dtype=np.float32)
