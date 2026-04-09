import logging
from pathlib import Path

from config import settings
from minio_client import minio_client


logger = logging.getLogger(__name__)


def ensure_model_exists_locally(model_name: str) -> Path:
    """Проверяет наличие модели локально, при необходимости скачивает из MinIO."""
    local_path = Path(settings.MODEL_CACHE_DIR) / model_name

    if local_path.exists():
        logger.info("Модель уже есть локально: %s", local_path)
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Скачиваю модель из MinIO: %s → %s", model_name, local_path)
    try:
        minio_client.fget_object(settings.MODEL_MINIO_BUCKET, model_name, str(local_path))
        logger.info("Модель скачана: %s", model_name)
    except Exception:
        logger.exception("Не удалось скачать модель %s", model_name)
        raise

    return local_path


def ensure_easyocr_weights_exists_locally() -> Path:
    """Скачивает веса EasyOCR (craft + языковые модели) из MinIO."""
    weights_dir = Path(settings.MODEL_CACHE_DIR) / settings.EASYOCR_WEIGHTS_DIR

    ocr_files = [
        "craft_mlt_25k.pth",
        "cyrillic_g2.pth",
        "english_g2.pth",
    ]

    weights_dir.mkdir(parents=True, exist_ok=True)

    for fname in ocr_files:
        local_path = weights_dir / fname
        if local_path.exists():
            continue

        minio_key = f"{settings.EASYOCR_WEIGHTS_DIR}/{fname}"
        logger.info("Скачиваю EasyOCR: %s → %s", minio_key, local_path)
        try:
            minio_client.fget_object(settings.MODEL_MINIO_BUCKET, minio_key, str(local_path))
        except Exception:
            logger.exception("Не удалось скачать %s", minio_key)
            raise

    return weights_dir


def load_models() -> bool:
    """Скачивает все модели из MinIO. Возвращает True при успехе."""
    try:
        ensure_model_exists_locally(settings.YOLO_MODEL_PATH)
        ensure_easyocr_weights_exists_locally()
        ensure_model_exists_locally(settings.BERT_MODEL_PATH)
        logger.info("Все модели загружены")
        return True
    except Exception:
        logger.exception("Ошибка при загрузке моделей")
        return False
