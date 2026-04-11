import logging
from pathlib import Path

import numpy as np
from config import settings
from PIL import Image
from ultralytics import YOLO


logger = logging.getLogger(__name__)

_yolo_model = None
_yolo_unavailable = False

MAX_DETECTIONS = 3


class YOLOModelNotFoundError(FileNotFoundError):
    pass


def get_yolo_model() -> YOLO | None:
    """Возвращает загруженную YOLO модель (singleton). None если недоступна."""
    global _yolo_model, _yolo_unavailable

    if _yolo_unavailable:
        return None

    if _yolo_model is not None:
        return _yolo_model

    try:
        model_path = Path(settings.MODEL_CACHE_DIR) / settings.YOLO_MODEL_PATH
        if not model_path.exists():
            raise YOLOModelNotFoundError(f"Модель YOLO не найдена: {model_path}")

        logger.info("Загрузка YOLO модели: %s", model_path)
        _yolo_model = YOLO(str(model_path))
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("YOLO недоступен (%s). Детекция пропускается.", e)
        _yolo_unavailable = True
        return None

    return _yolo_model


def detect_objects(image_path: str, confidence: float = 0.35) -> list[dict]:
    """Детектирует объекты на изображении. Возвращает топ-3 по confidence."""
    model = get_yolo_model()
    if model is None:
        logger.info("YOLO недоступен — возвращаю пустой список детекций.")
        return []

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    # Убираем альфа-канал если есть
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    h, w = img_array.shape[:2]

    logger.info("YOLO детекция: %dx%d, threshold=%.2f", w, h, confidence)
    results = model.predict(img_array, conf=confidence, device=settings.DEVICE, verbose=False)

    detections = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            try:
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(box.conf[0])

                # Нормализуем bbox к [0, 1]
                xyxy = box.xyxy[0].cpu().numpy()
                norm_bbox = [
                    round(float(xyxy[0]) / w, 4),
                    round(float(xyxy[1]) / h, 4),
                    round(float(xyxy[2]) / w, 4),
                    round(float(xyxy[3]) / h, 4),
                ]

                detections.append({
                    "class": cls_name,
                    "bbox": norm_bbox,
                    "confidence": round(conf, 4),
                })
            except Exception:
                logger.exception("Ошибка обработки бокса")

    # Сортируем по confidence и берём топ-3
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    detections = detections[:MAX_DETECTIONS]

    logger.info("YOLO: найдено %d объектов", len(detections))
    return detections
