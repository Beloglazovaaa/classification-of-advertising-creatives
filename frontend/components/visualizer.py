import logging
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image


logger = logging.getLogger(__name__)

BBOX_COORDINATES = 4


class InvalidImageSourceError(ValueError):
    pass


class ImageLoadingError(RuntimeError):
    def __init__(self, img_source, error):
        super().__init__(f"Ошибка загрузки изображения {img_source}: {error}")


def draw_bounding_boxes(
    image_path_or_url=None,
    ocr_blocks=None,
    detected_objects=None,
    ocr_color=(0, 255, 0),
    obj_color=(0, 255, 255),
):
    """Отрисовывает bounding boxes OCR и детекции на изображении."""
    if ocr_blocks is None:
        ocr_blocks = []
    if detected_objects is None:
        detected_objects = []

    if not image_path_or_url:
        raise InvalidImageSourceError("Не указан путь или URL изображения")

    try:
        if image_path_or_url.startswith(("http://", "https://")):
            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        img_array = np.array(image)
        h, w, _ = img_array.shape
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    except requests.RequestException as e:
        raise ImageLoadingError(image_path_or_url, e) from e

    # OCR блоки (зелёные)
    for block in ocr_blocks:
        bbox = block.get("bbox")
        if not bbox or len(bbox) != BBOX_COORDINATES:
            continue
        x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), ocr_color, 2)

        confidence = block.get("confidence", 0.0)
        _draw_label(img_cv, f"OCR {confidence:.2f}", (x1, y1), bg_color=ocr_color, text_color=(0, 0, 0))

    # Детектированные объекты (жёлтые)
    for obj in detected_objects:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != BBOX_COORDINATES:
            continue
        x1, y1, x2, y2 = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), obj_color, 2)

        class_name = obj.get("class", "unknown")
        confidence = obj.get("confidence", 0.0)
        _draw_label(img_cv, f"{class_name} {confidence:.2f}", (x1, y1), bg_color=obj_color, text_color=(0, 0, 0))

    img_with_boxes = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_with_boxes)


def _draw_label(image_cv, text, position, bg_color, text_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    padding = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    x, y = position

    text_y = max(y - 10, text_height + padding)

    cv2.rectangle(
        image_cv,
        (x, text_y - text_height - padding),
        (x + text_width + 2 * padding, text_y + padding),
        bg_color,
        -1,
    )

    cv2.putText(
        image_cv, text,
        (x + padding, text_y),
        font, font_scale, text_color, font_thickness, cv2.LINE_AA,
    )
