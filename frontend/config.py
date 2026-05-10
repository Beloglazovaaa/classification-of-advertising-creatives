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
    "bags":       "Сумки",
    "chairs":     "Стулья",
    "clocks":     "Часы (наст.)",
    "cups":       "Кружки",
    "cutlery":    "Ст. приборы",
    "glasses":    "Очки",
    "hats":       "Шляпы",
    "headphones": "Наушники",
    "jackets":    "Куртки",
    "lipsticks":  "Помады",
    "notebooks":  "Тетради",
    "perfumes":   "Парфюм",
    "phones":     "Телефоны",
    "shirts":     "Футболки",
    "shoes":      "Обувь",
    "socks":      "Носки",
    "ties":       "Галстуки",
    "umbrellas":  "Зонты",
    "wallets":    "Кошельки",
    "watches":    "Часы (нар.)",
}

COLOR_VISUAL_CLASSES = {
    # Ахроматические
    "Черный":          {"000000"},
    "Темно-серый":     {"404040"},
    "Серый":           {"808080"},
    "Светло-серый":    {"c8c8c8"},
    "Белый":           {"fafafa"},
    "Кремовый":        {"faf0dc"},
    "Слоновая кость":  {"fffaeb"},
    # Красные
    "Красный":         {"dc141e"},
    "Алый":            {"f03228"},
    "Темно-красный":   {"780a14"},
    "Бордовый":        {"821e37"},
    "Вишневый":        {"aa283c"},
    "Малиновый":       {"d21e50"},
    "Коралловый":      {"f0786e"},
    # Розовые
    "Розовый":         {"ff82aa"},
    "Фуксия":          {"ff3ca0"},
    "Пыльно-розовый":  {"c88c96"},
    "Пудровый":        {"e6c8c3"},
    "Лососевый":       {"fa8c78"},
    # Оранжевые
    "Оранжевый":       {"fa8c1e"},
    "Персиковый":      {"ffc8a0"},
    "Морковный":       {"f06e32"},
    "Янтарный":        {"e6aa28"},
    # Желтые
    "Желтый":          {"fadc28"},
    "Лимонный":        {"faf050"},
    "Горчичный":       {"cda528"},
    "Золотой":         {"dcb432"},
    "Песочный":        {"e6cd9b"},
    # Зеленые
    "Зеленый":         {"28aa3c"},
    "Темно-зеленый":   {"145a1e"},
    "Салатовый":       {"a0e65a"},
    "Оливковый":       {"828228"},
    "Мятный":          {"aae6c8"},
    "Хаки":            {"a0965f"},
    "Изумрудный":      {"1e8c6e"},
    # Бирюзовые
    "Бирюзовый":       {"28c8be"},
    "Аквамарин":       {"78dcc8"},
    # Голубые / Синие
    "Голубой":         {"50b4e6"},
    "Небесный":        {"96d2f0"},
    "Темно-голубой":   {"145a82"},
    "Синий":           {"1e3cd2"},
    "Темно-синий":     {"0f1e6e"},
    "Кобальт":         {"1e46b4"},
    "Индиго":          {"4b3296"},
    # Фиолетовые
    "Фиолетовый":      {"8228c8"},
    "Сиреневый":       {"b482e6"},
    "Маджента":        {"c81ea0"},
    "Лавандовый":      {"d2bee6"},
    "Баклажановый":    {"50284b"},
    # Коричневые / Бежевые
    "Коричневый":      {"6e461e"},
    "Шоколадный":      {"462814"},
    "Кофейный":        {"5f412d"},
    "Карамельный":     {"b4783c"},
    "Бежевый":         {"dcc8a0"},
    "Телесный":        {"f0d2b9"},
    "Молочный":        {"f5ebdc"},
}
