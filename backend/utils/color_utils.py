import logging
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

RESIZE_DIM = 150
SATURATION_THRESHOLD = 0.15
DARK_VALUE_THRESHOLD = 0.15


def get_top_colors(image_path: str, n_colors: int = 3) -> list[dict]:
    """Извлекает топ-N доминирующих цветов через KMeans."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((RESIZE_DIM, RESIZE_DIM))

    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    total_pixels = len(labels)

    # Сортируем по частоте
    sorted_indices = np.argsort(-counts)
    colors = []

    for idx in sorted_indices:
        r, g, b = centers[idx]
        percent = counts[idx] / total_pixels * 100
        hex_color = f"#{r:02x}{g:02x}{b:02x}"

        colors.append({
            "hex": hex_color,
            "rgb": [int(r), int(g), int(b)],
            "percent": round(float(percent), 1),
        })

    return colors


def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Конвертирует RGB [0-255] в HSV [H: 0-360, S: 0-1, V: 0-1]."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    c_max = max(r_norm, g_norm, b_norm)
    c_min = min(r_norm, g_norm, b_norm)
    delta = c_max - c_min

    # Hue
    if delta == 0:
        h = 0.0
    elif c_max == r_norm:
        h = 60.0 * (((g_norm - b_norm) / delta) % 6)
    elif c_max == g_norm:
        h = 60.0 * (((b_norm - r_norm) / delta) + 2)
    else:
        h = 60.0 * (((r_norm - g_norm) / delta) + 4)

    s = 0.0 if c_max == 0 else delta / c_max
    v = c_max

    return h, s, v


def _find_closest_monochrome_color(h: float, s: float, v: float) -> str:
    """Определяет ахроматический цвет по яркости."""
    if v <= DARK_VALUE_THRESHOLD:
        return "Черный"
    if v <= 0.3:
        return "Темно-серый"
    if v <= 0.6:
        return "Серый"
    if v <= 0.85:
        return "Светло-серый"
    return "Белый"


def _find_closest_chromatic_color(h: float, s: float, v: float) -> str:
    """Определяет хроматический цвет по оттенку."""
    # Красный (обёртка через 0/360)
    if h < 10 or h >= 350:
        return "Красный"
    if h < 25:
        if v < 0.5:
            return "Коричневый"
        return "Оранжевый"
    if h < 40:
        return "Оранжевый"
    if h < 70:
        return "Желтый"
    if h < 165:
        return "Зеленый"
    if h < 195:
        if v < 0.5:
            return "Темно-голубой"
        return "Голубой"
    if h < 250:
        return "Синий"
    if h < 290:
        return "Фиолетовый"
    if h < 310:
        return "Сиреневый"
    if h < 330:
        return "Маджента"
    if h < 350:
        return "Розовый"
    return "Красный"


def classify_colors_by_palette(colors: list[dict]) -> dict:
    """Классифицирует список цветов по палитре. Возвращает dict[class_name → {hex, percent}]."""
    classified = defaultdict(lambda: {"hex": "", "percent": 0.0})

    for color in colors:
        rgb = color.get("rgb", [128, 128, 128])
        r, g, b = rgb
        h, s, v = _rgb_to_hsv(r, g, b)

        if s < SATURATION_THRESHOLD:
            cls_name = _find_closest_monochrome_color(h, s, v)
        else:
            cls_name = _find_closest_chromatic_color(h, s, v)

        classified[cls_name]["percent"] += color.get("percent", 0.0)
        if not classified[cls_name]["hex"]:
            classified[cls_name]["hex"] = color.get("hex", "#808080")

    # Округляем
    result = {}
    for cls_name, info in classified.items():
        result[cls_name] = {
            "hex": info["hex"],
            "percent": round(info["percent"], 1),
        }

    return result
