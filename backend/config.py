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

# ── Категории товаров (20 классов) ─────────────────────────────────────
PRODUCT_TOPICS = {
    "bags":       {"ru": "Сумки",            "description": "Сумки, рюкзаки, клатчи"},
    "chairs":     {"ru": "Стулья",           "description": "Стулья, кресла"},
    "clocks":     {"ru": "Часы",             "description": "Настенные, настольные и наручные часы"},
    "cups":       {"ru": "Кружки",           "description": "Кружки, чашки, термосы"},
    "cutlery":    {"ru": "Столовые приборы", "description": "Ножи, вилки, ложки"},
    "glasses":    {"ru": "Очки",             "description": "Очки солнцезащитные и оптические"},
    "hats":       {"ru": "Шляпы",            "description": "Кепки, шляпы, головные уборы"},
    "headphones": {"ru": "Наушники",         "description": "Проводные и беспроводные наушники"},
    "jackets":    {"ru": "Куртки",           "description": "Куртки, пальто, бомберы"},
    "lipsticks":  {"ru": "Помады",           "description": "Губные помады и блески"},
    "notebooks":  {"ru": "Тетради",          "description": "Блокноты, тетради, ежедневники"},
    "perfumes":   {"ru": "Парфюм",           "description": "Парфюмерия, духи, туалетная вода"},
    "phones":     {"ru": "Телефоны",         "description": "Смартфоны и мобильные телефоны"},
    "shirts":     {"ru": "Футболки",         "description": "Футболки, рубашки, поло"},
    "shoes":      {"ru": "Обувь",            "description": "Кроссовки, туфли, ботинки"},
    "socks":      {"ru": "Носки",            "description": "Носки, гольфы"},
    "ties":       {"ru": "Галстуки",         "description": "Галстуки, бабочки"},
    "umbrellas":  {"ru": "Зонты",            "description": "Зонты-трости и складные"},
    "wallets":    {"ru": "Кошельки",         "description": "Кошельки, портмоне, картхолдеры"},
    "watches":    {"ru": "Часы",             "description": "Наручные часы и смарт-часы"},
}

TOPIC_LABELS = list(PRODUCT_TOPICS.keys())

# Маппинг имени папки датасета (singular) → topic-label из TOPIC_LABELS
DATASET_FOLDER_TO_TOPIC = {
    "bag": "bags",
    "chair": "chairs",
    "clock": "clocks",
    "cup": "cups",
    "cutlery": "cutlery",
    "glasses": "glasses",
    "hat": "hats",
    "headphones": "headphones",
    "jacket": "jackets",
    "lipstick": "lipsticks",
    "notebook": "notebooks",
    "perfume": "perfumes",
    "phone": "phones",
    "shirt": "shirts",
    "shoes": "shoes",
    "socks": "socks",
    "tie": "ties",
    "umbrella": "umbrellas",
    "wallet": "wallets",
    "watch": "watches",
}

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
    # cutlery
    "fork": "cutlery", "knife": "cutlery", "spoon": "cutlery",
    # ties
    "tie": "ties",
    # bags
    "backpack": "bags", "handbag": "bags", "suitcase": "bags",
    # cups
    "cup": "cups", "bowl": "cups", "wine glass": "cups", "bottle": "cups",
    # clocks
    "clock": "clocks",
    # chairs
    "chair": "chairs", "couch": "chairs", "bench": "chairs",
    # umbrellas
    "umbrella": "umbrellas",
    # notebooks
    "book": "notebooks",
    # glasses, socks — нет прямого COCO-класса, fallback не сработает
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
    # Ахроматические
    "Черный":          "#000000",
    "Темно-серый":     "#404040",
    "Серый":           "#808080",
    "Светло-серый":    "#c8c8c8",
    "Белый":           "#fafafa",
    "Кремовый":        "#faf0dc",
    "Слоновая кость":  "#fffaeb",
    # Красные
    "Красный":         "#dc141e",
    "Алый":            "#f03228",
    "Темно-красный":   "#780a14",
    "Бордовый":        "#821e37",
    "Вишневый":        "#aa283c",
    "Малиновый":       "#d21e50",
    "Коралловый":      "#f0786e",
    # Розовые
    "Розовый":         "#ff82aa",
    "Фуксия":          "#ff3ca0",
    "Пыльно-розовый":  "#c88c96",
    "Пудровый":        "#e6c8c3",
    "Лососевый":       "#fa8c78",
    # Оранжевые
    "Оранжевый":       "#fa8c1e",
    "Персиковый":      "#ffc8a0",
    "Морковный":       "#f06e32",
    "Янтарный":        "#e6aa28",
    # Желтые
    "Желтый":          "#fadc28",
    "Лимонный":        "#faf050",
    "Горчичный":       "#cda528",
    "Золотой":         "#dcb432",
    "Песочный":        "#e6cd9b",
    # Зеленые
    "Зеленый":         "#28aa3c",
    "Темно-зеленый":   "#145a1e",
    "Салатовый":       "#a0e65a",
    "Оливковый":       "#828228",
    "Мятный":          "#aae6c8",
    "Хаки":            "#a0965f",
    "Изумрудный":      "#1e8c6e",
    # Бирюзовые
    "Бирюзовый":       "#28c8be",
    "Аквамарин":       "#78dcc8",
    # Голубые / Синие
    "Голубой":         "#50b4e6",
    "Небесный":        "#96d2f0",
    "Темно-голубой":   "#145a82",
    "Синий":           "#1e3cd2",
    "Темно-синий":     "#0f1e6e",
    "Кобальт":         "#1e46b4",
    "Индиго":          "#4b3296",
    # Фиолетовые
    "Фиолетовый":      "#8228c8",
    "Сиреневый":       "#b482e6",
    "Маджента":        "#c81ea0",
    "Лавандовый":      "#d2bee6",
    "Баклажановый":    "#50284b",
    # Коричневые / Бежевые
    "Коричневый":      "#6e461e",
    "Шоколадный":      "#462814",
    "Кофейный":        "#5f412d",
    "Карамельный":     "#b4783c",
    "Бежевый":         "#dcc8a0",
    "Телесный":        "#f0d2b9",
    "Молочный":        "#f5ebdc",
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
