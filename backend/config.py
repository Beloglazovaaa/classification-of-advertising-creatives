import numpy as np
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@db:5432/creatives_db"
    REDIS_URL: str = "redis://redis:6379/0"
    UPLOAD_FOLDER: str = "./uploads"

    MINIO_ENDPOINT: str = "minio:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET: str = "creatives"
    MINIO_PUBLIC_URL: str = "http://localhost:9000"

    MODEL_CACHE_DIR: str = "/app/models"
    MODEL_MINIO_BUCKET: str = "models"
    YOLO_MODEL_PATH: str = "yolov8m.pt"
    EASYOCR_WEIGHTS_DIR: str = "easy_ocr"
    BERT_MODEL_PATH: str = "best_multimodal_bert.pt"
    DEVICE: str = "cpu"

    CELERY_CONCURRENCY: int = 2

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# ── Категории товаров ──────────────────────────────────────────────────
PRODUCT_TOPICS = {
    "cutlery": {
        "ru": "Столовые приборы",
        "description": "Ножи, вилки, ложки и наборы столовых приборов",
    },
    "ties": {
        "ru": "Галстуки",
        "description": "Мужские галстуки, бабочки и аксессуары",
    },
    "bags": {
        "ru": "Сумки",
        "description": "Сумки, рюкзаки, клатчи и портфели",
    },
    "cups": {
        "ru": "Кружки",
        "description": "Кружки, чашки, термосы и стаканы",
    },
    "clocks": {
        "ru": "Часы",
        "description": "Настенные, наручные и настольные часы",
    },
}

TOPIC_LABELS = list(PRODUCT_TOPICS.keys())

# ── COCO классы для YOLO ───────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

NUM_COCO = len(COCO_CLASSES)
COCO_CLASS_TO_IDX = {cls: i for i, cls in enumerate(COCO_CLASSES)}

YOLO_CONFIDENCE_THRESHOLD = 0.35

# ── Маппинг COCO-класс → топик продукта ───────────────────────────────
_COCO_TOPIC_MAP = {
    "fork": "cutlery", "knife": "cutlery", "spoon": "cutlery",
    "tie": "ties",
    "backpack": "bags", "handbag": "bags", "suitcase": "bags",
    "cup": "cups", "bowl": "cups", "wine glass": "cups", "bottle": "cups",
    "clock": "clocks",
}


def map_coco_to_topic(coco_class: str) -> str | None:
    return _COCO_TOPIC_MAP.get(coco_class)


# ── Цветовая палитра ──────────────────────────────────────────────────
COLOR_CLASSES = {
    "Красный": [
        (0, 0.7, 0.4), (10, 1.0, 1.0), (350, 1.0, 1.0), (360, 0.7, 0.4),
    ],
    "Коричневый": [(10, 0.3, 0.2), (30, 0.8, 0.5)],
    "Розовый": [(300, 0.2, 0.7), (350, 0.6, 1.0)],
    "Оранжевый": [(15, 0.7, 0.6), (35, 1.0, 1.0)],
    "Желтый": [(40, 0.5, 0.6), (65, 1.0, 1.0)],
    "Зеленый": [(70, 0.3, 0.3), (165, 1.0, 1.0)],
    "Голубой": [(170, 0.3, 0.5), (195, 1.0, 1.0)],
    "Темно-голубой": [(170, 0.3, 0.2), (195, 0.8, 0.5)],
    "Синий": [(200, 0.3, 0.3), (250, 1.0, 1.0)],
    "Фиолетовый": [(250, 0.3, 0.3), (290, 1.0, 1.0)],
    "Маджента": [(290, 0.5, 0.5), (330, 1.0, 1.0)],
    "Сиреневый": [(270, 0.2, 0.5), (310, 0.5, 0.8)],
    "Черный": "mono_dark",
    "Темно-серый": "mono_dark_gray",
    "Серый": "mono_gray",
    "Светло-серый": "mono_light_gray",
    "Белый": "mono_white",
}

COLOR_VISUAL_HEX = {
    "Красный": "#bc0e0e",
    "Коричневый": "#663300",
    "Розовый": "#ff0080",
    "Оранжевый": "#f27900",
    "Желтый": "#f2f20c",
    "Зеленый": "#009900",
    "Голубой": "#29cccc",
    "Темно-голубой": "#008080",
    "Синий": "#0a4bcc",
    "Фиолетовый": "#7e17e5",
    "Маджента": "#ff00ff",
    "Сиреневый": "#b300b3",
    "Черный": "#000000",
    "Темно-серый": "#404040",
    "Серый": "#808080",
    "Светло-серый": "#bfbfbf",
    "Белый": "#f7f7f7",
}

# ── Настройки цветового анализа ────────────────────────────────────────
DOMINANT_COLORS_COUNT = 3
SECONDARY_COLORS_COUNT = 3

# ── Стадии ML-пайплайна ───────────────────────────────────────────────
ML_STAGES = [
    {"name": "ocr", "status_field": "ocr_status", "start_field": "ocr_start", "duration_field": "ocr_duration"},
    {"name": "detection", "status_field": "detection_status", "start_field": "detection_start", "duration_field": "detection_duration"},
    {"name": "classification", "status_field": "classification_status", "start_field": "classification_start", "duration_field": "classification_duration"},
    {"name": "color_analysis", "status_field": "color_status", "start_field": "color_start", "duration_field": "color_duration"},
]
