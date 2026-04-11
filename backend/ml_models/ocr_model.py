import logging
from pathlib import Path

import easyocr
from config import settings


logger = logging.getLogger(__name__)

_ocr_reader = None
_ocr_unavailable = False


class EasyOCRModelDirNotFoundError(FileNotFoundError):
    pass


def get_ocr_reader() -> easyocr.Reader | None:
    """Возвращает инициализированный EasyOCR reader (singleton). None если недоступен."""
    global _ocr_reader, _ocr_unavailable

    if _ocr_unavailable:
        return None

    if _ocr_reader is not None:
        return _ocr_reader

    try:
        model_dir = Path(settings.MODEL_CACHE_DIR) / settings.EASYOCR_WEIGHTS_DIR
        if not model_dir.exists():
            raise EasyOCRModelDirNotFoundError(
                f"Директория весов EasyOCR не найдена: {model_dir}",
            )

        gpu = settings.DEVICE != "cpu"
        logger.info("Инициализация EasyOCR (GPU=%s, dir=%s)", gpu, model_dir)

        _ocr_reader = easyocr.Reader(
            ["en", "ru"],
            gpu=gpu,
            model_storage_directory=str(model_dir),
            download_enabled=False,
        )
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("EasyOCR недоступен (%s). OCR пропускается.", e)
        _ocr_unavailable = True
        return None

    return _ocr_reader


def extract_text_and_blocks(image_path: str, creative) -> tuple[str, list[dict]]:
    """Извлекает текст и блоки с координатами из изображения."""
    reader = get_ocr_reader()
    if reader is None:
        logger.info("OCR недоступен — возвращаю пустой результат.")
        return "", []

    results = reader.readtext(image_path)

    if not results:
        return "", []

    texts = []
    blocks = []

    w = creative.image_width or 1
    h = creative.image_height or 1

    for bbox, text, confidence in results:
        texts.append(text)

        # Нормализуем bbox к [0, 1]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        norm_bbox = [
            min(xs) / w,
            min(ys) / h,
            max(xs) / w,
            max(ys) / h,
        ]

        blocks.append({
            "text": text,
            "bbox": [round(x, 4) for x in norm_bbox],
            "confidence": round(float(confidence), 4),
        })

    full_text = " ".join(texts)
    logger.info("OCR: найдено %d блоков текста", len(blocks))

    return full_text, blocks
