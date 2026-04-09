import os

from dotenv import load_dotenv


load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "creatives")

if MINIO_SECURE:
    MINIO_BASE_URL = f"https://{MINIO_ENDPOINT}"
else:
    MINIO_BASE_URL = f"http://{MINIO_ENDPOINT}"

MINIO_PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL") or MINIO_BASE_URL

THUMBNAIL_WIDTH = 120
ESTIMATED_CONTENT_WIDTH = 1000
MAX_COLUMNS = 10
MIN_COLUMNS = 1

TOPIC_TRANSLATIONS = {
    "cutlery": "Ст. приборы",
    "ties": "Галстуки",
    "bags": "Сумки",
    "cups": "Кружки",
    "clocks": "Часы",
}

COLOR_VISUAL_CLASSES = {
    "Красный": {"bc0e0e"},
    "Коричневый": {"663300"},
    "Розовый": {"ff0080"},
    "Оранжевый": {"f27900"},
    "Желтый": {"f2f20c"},
    "Зеленый": {"009900"},
    "Голубой": {"29cccc"},
    "Темно-голубой": {"008080"},
    "Синий": {"0a4bcc"},
    "Фиолетовый": {"7e17e5"},
    "Маджента": {"ff00ff"},
    "Сиреневый": {"b300b3"},
    "Черный": {"000000"},
    "Темно-серый": {"404040"},
    "Серый": {"808080"},
    "Светло-серый": {"bfbfbf"},
    "Белый": {"f7f7f7"},
}
