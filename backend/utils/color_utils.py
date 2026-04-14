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

# Внутренний размер кластеризации: даже если пользователь просит 3 цвета,
# KMeans всегда строит как минимум MIN_INTERNAL_CLUSTERS центров, а сверху
# возвращаются топ-N. Это важно для картинок с множеством объектов, где при
# k=3 предметы сливаются с фоном в один кластер.
MIN_INTERNAL_CLUSTERS = 12

# Сила центровой маски (гауссиан). Чем больше — тем сильнее приоритет
# центральных пикселей над угловыми. 0.0 выключает маску полностью.
CENTER_WEIGHT_SIGMA = 0.35  # доля от половины стороны изображения
CENTER_WEIGHT_MIN = 0.15    # минимальный вес любого пикселя (чтобы не занулять фон)

# Порог тёмности (в L канале Lab, диапазон 0..100)
DARK_L_THRESHOLD = 18.0
# Порог "почти серый" — низкая хроматичность (sqrt(a²+b²)).
# 12.0 оказалось слишком агрессивно: тёплые пастельные фоны (пудровый,
# бежевый, телесный) имеют chroma ≈ 8..12 и ошибочно попадали в серый ряд.
# 5.0 оставляет в ахроматическом классе только действительно нейтральные
# оттенки (фотобумага, белые стены, тёмные фоны студии).
ACHROMATIC_C_THRESHOLD = 5.0


# ───────────────────────── Опорная палитра ─────────────────────────
# Каждое имя → RGB. Lab вычислим один раз ниже.
PALETTE_RGB: dict[str, tuple[int, int, int]] = {
    # ─── Ахроматические ───
    "Черный":          (0, 0, 0),
    "Темно-серый":     (64, 64, 64),
    "Серый":           (128, 128, 128),
    "Светло-серый":    (200, 200, 200),
    "Белый":           (250, 250, 250),
    "Кремовый":        (250, 240, 220),
    "Слоновая кость":  (255, 250, 235),
    # ─── Красные ───
    "Красный":         (220, 20, 30),
    "Алый":            (240, 50, 40),
    "Темно-красный":   (120, 10, 20),
    "Бордовый":        (130, 30, 55),
    "Вишневый":        (170, 40, 60),
    "Малиновый":       (210, 30, 80),
    "Коралловый":      (240, 120, 110),
    # ─── Розовые ───
    "Розовый":         (255, 130, 170),
    "Фуксия":          (255, 60, 160),
    "Пыльно-розовый":  (200, 140, 150),
    "Пудровый":        (230, 200, 195),
    "Лососевый":       (250, 140, 120),
    # ─── Оранжевые ───
    "Оранжевый":       (250, 140, 30),
    "Персиковый":      (255, 200, 160),
    "Морковный":       (240, 110, 50),
    "Янтарный":        (230, 170, 40),
    # ─── Желтые ───
    "Желтый":          (250, 220, 40),
    "Лимонный":        (250, 240, 80),
    "Горчичный":       (205, 165, 40),
    "Золотой":         (220, 180, 50),
    "Песочный":        (230, 205, 155),
    # ─── Зеленые ───
    "Зеленый":         (40, 170, 60),
    "Темно-зеленый":   (20, 90, 30),
    "Салатовый":       (160, 230, 90),
    "Оливковый":       (130, 130, 40),
    "Мятный":          (170, 230, 200),
    "Хаки":            (160, 150, 95),
    "Изумрудный":      (30, 140, 110),
    # ─── Бирюзовые ───
    "Бирюзовый":       (40, 200, 190),
    "Аквамарин":       (120, 220, 200),
    # ─── Голубые / Синие ───
    "Голубой":         (80, 180, 230),
    "Небесный":        (150, 210, 240),
    "Темно-голубой":   (20, 90, 130),
    "Синий":           (30, 60, 210),
    "Темно-синий":     (15, 30, 110),
    "Кобальт":         (30, 70, 180),
    "Индиго":          (75, 50, 150),
    # ─── Фиолетовые ───
    "Фиолетовый":      (130, 40, 200),
    "Сиреневый":       (180, 130, 230),
    "Маджента":        (200, 30, 160),
    "Лавандовый":      (210, 190, 230),
    "Баклажановый":    (80, 40, 75),
    # ─── Коричневые / Бежевые ───
    "Коричневый":      (110, 70, 30),
    "Шоколадный":      (70, 40, 20),
    "Кофейный":        (95, 65, 45),
    "Карамельный":     (180, 120, 60),
    "Бежевый":         (220, 200, 160),
    "Телесный":        (240, 210, 185),
    "Молочный":        (245, 235, 220),
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


def _build_center_weights(h: int, w: int) -> np.ndarray:
    """Возвращает массив весов (h*w,), приоритетизирующий центральные пиксели.

    Используется как sample_weight для KMeans: у продуктовых фото предмет
    почти всегда в центре, а фон — по краям. Центровая маска позволяет
    «предметным» цветам пробиться в топ кластеров, не прибегая к отдельной
    модели сегментации.
    """
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist2 = xx * xx + yy * yy
    sigma = max(CENTER_WEIGHT_SIGMA, 1e-3)
    weights = np.exp(-dist2 / (2.0 * sigma * sigma))
    # Нормируем в [CENTER_WEIGHT_MIN, 1.0], чтобы фон всё же учитывался.
    weights = CENTER_WEIGHT_MIN + (1.0 - CENTER_WEIGHT_MIN) * weights
    return weights.reshape(-1)


def get_top_colors(image_path: str, n_colors: int = 3) -> list[dict]:
    """Извлекает топ-N доминирующих цветов через KMeans в Lab-пространстве.

    Возвращает список словарей: {hex, rgb, lab, percent} по убыванию percent.

    Внутри функция всегда строит расширенный набор кластеров
    (не меньше MIN_INTERNAL_CLUSTERS), а затем отдаёт топ-N по взвешенной доле.
    Пиксели весятся маской, приоритетизирующей центр изображения, — это
    уменьшает доминирование фона на продуктовых фото.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize((RESIZE_DIM, RESIZE_DIM), Image.Resampling.LANCZOS)

    rgb_pixels = np.array(image, dtype=np.uint8).reshape(-1, 3)
    lab_pixels = _rgb_to_lab(rgb_pixels)  # (N, 3)

    sample_weights = _build_center_weights(RESIZE_DIM, RESIZE_DIM)

    unique_count = len(np.unique(lab_pixels, axis=0))
    internal_k = max(n_colors, MIN_INTERNAL_CLUSTERS)
    n_clusters = max(1, min(internal_k, unique_count))
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=KMEANS_N_INIT,
        random_state=RANDOM_STATE,
    )
    kmeans.fit(lab_pixels, sample_weight=sample_weights)

    centers_lab = kmeans.cluster_centers_  # (k, 3) в Lab
    labels = kmeans.labels_
    # Вес кластера = сумма весов его пикселей (а не просто их число),
    # чтобы центральные предметные цвета получали больший вклад.
    counts = np.zeros(n_clusters, dtype=np.float64)
    np.add.at(counts, labels, sample_weights)
    total_pixels = float(counts.sum())

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

    sorted_idx = np.argsort(-counts)[:n_colors]
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
