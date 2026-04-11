"""Цветовой анализ изображений в перцептивном пространстве CIE Lab.

Почему Lab, а не RGB/HSV:
- Lab перцептивно равномерен: евклидово расстояние ≈ воспринимаемая разница цветов.
- KMeans на Lab даёт куда более естественные кластеры, чем на RGB.
- Классификация ближайшего цвета по Delta E (CIE76) — индустриальный стандарт.

Pipeline:
    image → resize → RGB → Lab → KMeans(k=n) → центры в Lab
    → центры в RGB (для UI) → классификация по словарю опорных Lab-цветов.
"""
import logging
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)

RESIZE_DIM = 200          # baseline разрешения для KMeans (быстрее и стабильнее)
KMEANS_N_INIT = 10
RANDOM_STATE = 42

# Порог тёмности (в L канале Lab, диапазон 0..100)
DARK_L_THRESHOLD = 18.0
# Порог "почти серый" — низкая хроматичность (sqrt(a²+b²))
ACHROMATIC_C_THRESHOLD = 12.0


# ───────────────────────── Опорная палитра ─────────────────────────
# Каждое имя → RGB. Lab вычислим один раз ниже.
PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    # Ахроматические
    "Черный":        (0, 0, 0),
    "Темно-серый":   (64, 64, 64),
    "Серый":         (128, 128, 128),
    "Светло-серый":  (200, 200, 200),
    "Белый":         (255, 255, 255),
    # Хроматические
    "Красный":       (220, 20, 30),
    "Темно-красный": (120, 10, 20),
    "Розовый":       (255, 130, 170),
    "Маджента":      (200, 30, 160),
    "Фиолетовый":    (130, 40, 200),
    "Сиреневый":     (180, 130, 230),
    "Синий":         (30, 60, 210),
    "Голубой":       (80, 180, 230),
    "Темно-голубой": (20, 90, 130),
    "Бирюзовый":     (40, 200, 190),
    "Зеленый":       (40, 170, 60),
    "Темно-зеленый": (20, 90, 30),
    "Салатовый":     (160, 230, 90),
    "Желтый":        (250, 220, 40),
    "Оранжевый":     (250, 140, 30),
    "Коричневый":    (110, 70, 30),
    "Бежевый":       (220, 200, 160),
}


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB [0..255] (H,W,3) или (N,3) → Lab (тех же форм)."""
    arr = rgb.astype(np.uint8)
    if arr.ndim == 2:  # (N,3)
        arr = arr.reshape(-1, 1, 3)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    else:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV: L∈[0,255], a,b∈[0,255] (offset 128). Приводим к стандартным:
    # L∈[0,100], a,b∈[-128,127]
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab


# Pre-compute опорной палитры в Lab (один раз при импорте)
_PALETTE_NAMES: list[str] = list(PALETTE_RGB.keys())
_PALETTE_LAB: np.ndarray = _rgb_to_lab(
    np.array(list(PALETTE_RGB.values()), dtype=np.uint8),
)


def get_top_colors(image_path: str, n_colors: int = 3) -> list[dict]:
    """Извлекает топ-N доминирующих цветов через KMeans в Lab-пространстве.

    Возвращает список словарей: {hex, rgb, lab, percent} по убыванию percent.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((RESIZE_DIM, RESIZE_DIM), Image.Resampling.LANCZOS)

    rgb_pixels = np.array(image, dtype=np.uint8).reshape(-1, 3)
    lab_pixels = _rgb_to_lab(rgb_pixels)  # (N, 3)

    n_clusters = max(1, min(n_colors, len(np.unique(lab_pixels, axis=0))))
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    kmeans.fit(lab_pixels)

    centers_lab = kmeans.cluster_centers_  # (k, 3) в Lab
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=n_clusters)
    total_pixels = float(len(labels))

    # Конвертируем центры обратно в RGB для UI
    # Сначала Lab→OpenCV-Lab, затем cvtColor LAB2RGB
    centers_cv_lab = centers_lab.copy()
    centers_cv_lab[:, 0] = centers_cv_lab[:, 0] * (255.0 / 100.0)
    centers_cv_lab[:, 1] = centers_cv_lab[:, 1] + 128.0
    centers_cv_lab[:, 2] = centers_cv_lab[:, 2] + 128.0
    centers_cv_lab = np.clip(centers_cv_lab, 0, 255).astype(np.uint8)
    centers_rgb = cv2.cvtColor(
        centers_cv_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB,
    ).reshape(-1, 3)

    sorted_idx = np.argsort(-counts)
    colors = []
    for idx in sorted_idx:
        r, g, b = (int(x) for x in centers_rgb[idx])
        L, a_, b_ = (float(x) for x in centers_lab[idx])
        percent = counts[idx] / total_pixels * 100.0
        colors.append({
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "rgb": [r, g, b],
            "lab": [round(L, 2), round(a_, 2), round(b_, 2)],
            "percent": round(float(percent), 1),
        })
    return colors


def _classify_lab(L: float, a: float, b: float) -> str:
    """Возвращает имя ближайшего цвета палитры по Delta E (CIE76 в Lab)."""
    # Спецслучай: очень тёмное → "Черный" даже если палитра ближе к коричневому
    if L < DARK_L_THRESHOLD:
        return "Черный"

    # Низкая хроматичность → ахроматический ряд по L
    chroma = float(np.sqrt(a * a + b * b))
    if chroma < ACHROMATIC_C_THRESHOLD:
        if L < 25:
            return "Темно-серый"
        if L < 55:
            return "Серый"
        if L < 80:
            return "Светло-серый"
        return "Белый"

    # Иначе — ближайший по евклидову расстоянию в Lab
    point = np.array([L, a, b], dtype=np.float32)
    dists = np.linalg.norm(_PALETTE_LAB - point, axis=1)
    return _PALETTE_NAMES[int(np.argmin(dists))]


def classify_colors_by_palette(colors: list[dict]) -> dict:
    """Классифицирует список цветов по палитре через Delta E в Lab.

    Возвращает dict[class_name → {hex, percent}].
    """
    classified: dict[str, dict] = defaultdict(lambda: {"hex": "", "percent": 0.0})

    for color in colors:
        # Если есть готовый Lab — используем; иначе считаем из RGB.
        if "lab" in color:
            L, a, b = color["lab"]
        else:
            rgb = np.array([color.get("rgb", [128, 128, 128])], dtype=np.uint8)
            L, a, b = _rgb_to_lab(rgb)[0].tolist()

        cls_name = _classify_lab(float(L), float(a), float(b))
        classified[cls_name]["percent"] += color.get("percent", 0.0)
        if not classified[cls_name]["hex"]:
            classified[cls_name]["hex"] = color.get("hex", "#808080")

    return {
        cls_name: {"hex": info["hex"], "percent": round(info["percent"], 1)}
        for cls_name, info in classified.items()
    }
