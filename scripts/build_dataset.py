"""Сборка датасета креативов по категориям через DuckDuckGo Image Search.

Скачивает изображения по списку поисковых запросов в `dataset/raw/<category>/`
с консистентным неймингом `<category>_NN.<ext>`.

Использование:
    pip install ddgs pillow requests tqdm
    python scripts/build_dataset.py --per-category 25
    python scripts/build_dataset.py --only chair umbrella --per-category 30
    python scripts/build_dataset.py --csv dataset/train.csv  # сгенерировать CSV-лейблы
"""
from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import requests
from PIL import Image


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "raw"

# ──── Категории и поисковые запросы ────
# По несколько запросов на категорию → больше разнообразия и валидных результатов.
CATEGORIES: dict[str, list[str]] = {
    # ── Базовые 10 ──
    "bag":      ["fashion bag advertisement", "designer handbag banner", "leather bag ad"],
    "clock":    ["wall clock advertisement", "wristwatch banner ad", "luxury watch ad"],
    "cutlery":  ["cutlery set advertisement", "fork knife spoon banner", "kitchen cutlery ad"],
    "cup":      ["coffee mug advertisement", "ceramic cup banner", "tea cup ad"],
    "tie":      ["necktie advertisement", "men silk tie banner", "formal tie ad"],
    "glasses":  ["sunglasses advertisement", "eyeglasses banner ad", "designer glasses ad"],
    "chair":    ["office chair advertisement", "modern chair banner", "designer chair ad"],
    "socks":    ["socks advertisement", "colorful socks banner", "men socks ad"],
    "umbrella": ["umbrella advertisement", "rain umbrella banner", "designer umbrella ad"],
    "notebook": ["notebook advertisement", "paper notebook banner", "stationery notebook ad"],
    # ── Дополнительные 10 (одежда, обувь, аксессуары, дом) ──
    "shoes":    ["sneakers advertisement", "running shoes banner", "leather shoes ad"],
    "hat":      ["cap advertisement", "fedora hat banner", "winter hat ad"],
    "watch":    ["smartwatch advertisement", "fitness watch banner", "apple watch ad"],
    "wallet":   ["leather wallet advertisement", "men wallet banner", "card wallet ad"],
    "perfume":  ["perfume advertisement", "fragrance bottle banner", "luxury perfume ad"],
    "lipstick": ["lipstick advertisement", "lip cosmetic banner", "matte lipstick ad"],
    "phone":    ["smartphone advertisement", "iphone banner", "android phone ad"],
    "headphones":["headphones advertisement", "wireless earbuds banner", "audio headset ad"],
    "shirt":    ["t-shirt advertisement", "casual shirt banner", "polo shirt ad"],
    "jacket":   ["jacket advertisement", "winter coat banner", "leather jacket ad"],
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_BYTES = 5_000          # отбрасываем явно сломанные / 1×1 файлы
MIN_DIMENSION = 200        # минимальная сторона изображения в пикселях
DOWNLOAD_TIMEOUT = 15
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}


def _ddg_search(query: str, max_results: int) -> list[str]:
    """Возвращает список URL изображений по запросу через DDGS."""
    try:
        from ddgs import DDGS  # ddgs >= 6.0
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # старая версия
        except ImportError:
            logger.error(
                "Установи библиотеку: pip install ddgs (или duckduckgo-search)",
            )
            sys.exit(1)

    urls: list[str] = []
    with DDGS() as ddgs:
        try:
            results = ddgs.images(
                query=query,
                max_results=max_results,
                safesearch="moderate",
                size="Medium",
            )
            for r in results:
                u = r.get("image") or r.get("url")
                if u:
                    urls.append(u)
        except Exception as e:
            logger.warning("DDGS error для '%s': %s", query, e)
    return urls


def _download_one(url: str, dst_dir: Path, idx: int, category: str) -> bool:
    """Скачивает один URL, валидирует изображение, сохраняет в dst_dir."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=DOWNLOAD_TIMEOUT, stream=True)
        if r.status_code != 200:
            return False
        content = r.content
        if len(content) < MIN_BYTES:
            return False

        # Открываем через PIL для валидации и определения формата
        img = Image.open(io.BytesIO(content))
        img.verify()  # быстрая проверка целостности
        img = Image.open(io.BytesIO(content))  # верификация закрывает файл — открываем заново
        if min(img.size) < MIN_DIMENSION:
            return False

        fmt = (img.format or "JPEG").lower()
        ext = "jpg" if fmt == "jpeg" else fmt
        if f".{ext}" not in VALID_EXTS:
            return False

        # Конвертим всё в RGB jpeg для единообразия
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        dst_path = dst_dir / f"{category}_{idx:02d}.jpg"
        img.convert("RGB").save(dst_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def collect_category(category: str, queries: list[str], per_category: int, force: bool) -> int:
    """Собирает per_category изображений для одной категории."""
    dst_dir = RAW_DIR / category
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = list(dst_dir.glob(f"{category}_*.jpg"))
        if len(existing) >= per_category:
            logger.info("✓ %s: уже собрано %d (skip, --force чтобы перекачать)",
                        category, len(existing))
            return len(existing)

    logger.info("⬇ %s: цель = %d изображений", category, per_category)
    seen_urls: set[str] = set()
    candidates: list[str] = []

    # Собираем кандидатов по всем запросам
    for q in queries:
        # запрашиваем с запасом ×3, чтобы покрыть отбрасываемые
        urls = _ddg_search(q, per_category * 3 // len(queries) + 5)
        for u in urls:
            if u not in seen_urls:
                seen_urls.add(u)
                candidates.append(u)
        time.sleep(0.5)  # вежливая пауза

    logger.info("  кандидатов: %d", len(candidates))

    saved = 0
    idx = 1
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {}
        for url in candidates:
            if saved + len(futures) >= per_category * 2:
                break
            fut = ex.submit(_download_one, url, dst_dir, idx, category)
            futures[fut] = url
            idx += 1

        for fut in as_completed(futures):
            if fut.result():
                saved += 1
                if saved >= per_category:
                    break

    # Перенумеровываем чтобы был ровный порядок _01..NN
    files = sorted(dst_dir.glob(f"{category}_*.jpg"))
    for i, f in enumerate(files, start=1):
        target = dst_dir / f"{category}_{i:02d}.jpg"
        if f != target:
            f.rename(target)

    final_count = len(list(dst_dir.glob(f"{category}_*.jpg")))
    logger.info("✓ %s: сохранено %d", category, final_count)
    return final_count


# Маппинг имени папки → topic-label (должен совпадать с backend/config.py
# DATASET_FOLDER_TO_TOPIC)
FOLDER_TO_TOPIC: dict[str, str] = {
    "bag": "bags",
    "chair": "chairs",
    "clock": "clocks",
    "cup": "cups",
    "cutlery": "cutlery",
    "glasses": "glasses",
    "notebook": "notebooks",
    "socks": "socks",
    "tie": "ties",
    "umbrella": "umbrellas",
    "shoes": "shoes",
    "hat": "hats",
    "watch": "watches",
    "wallet": "wallets",
    "perfume": "perfumes",
    "lipstick": "lipsticks",
    "phone": "phones",
    "headphones": "headphones",
    "shirt": "shirts",
    "jacket": "jackets",
}


def write_csv(out_path: Path) -> None:
    """Создаёт train.csv с минимальным набором колонок для train_bert.py.

    Имя папки (singular) маппится в topic-label, чтобы совпадало с TOPIC_LABELS.
    """
    import csv

    rows = []
    for category in sorted(RAW_DIR.iterdir()):
        if not category.is_dir():
            continue
        topic = FOLDER_TO_TOPIC.get(category.name, category.name)
        for img in sorted(category.glob("*")):
            if img.suffix.lower() in VALID_EXTS:
                rows.append({
                    "image_path": str(img.relative_to(PROJECT_ROOT)),
                    "ocr_text": "",          # будет заполнено OCR-проходом
                    "yolo_classes": "",      # будет заполнено YOLO-проходом
                    "yolo_confs": "",
                    "topic": topic,
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_path", "ocr_text", "yolo_classes", "yolo_confs", "topic"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("📝 CSV сохранён: %s (%d строк)", out_path, len(rows))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Сборка датасета креативов")
    p.add_argument("--per-category", type=int, default=25,
                   help="Сколько изображений на категорию")
    p.add_argument("--only", nargs="*", default=None,
                   help="Скачивать только указанные категории")
    p.add_argument("--force", action="store_true",
                   help="Перекачать даже если уже собрано")
    p.add_argument("--csv", type=Path, default=None,
                   help="После сбора сгенерировать CSV-лейблы")
    p.add_argument("--csv-only", action="store_true",
                   help="Только создать CSV, без скачивания")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.csv_only:
        targets = args.only if args.only else list(CATEGORIES.keys())
        for cat in targets:
            if cat not in CATEGORIES:
                logger.warning("Неизвестная категория: %s", cat)
                continue
            collect_category(cat, CATEGORIES[cat], args.per_category, args.force)

    if args.csv or args.csv_only:
        out = args.csv or (PROJECT_ROOT / "dataset" / "train.csv")
        write_csv(out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
