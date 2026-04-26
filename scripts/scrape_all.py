"""Полный ребилд датасета через Bing (icrawler).

Качает по ~60 картинок на каждый из 20 классов с конкретными запросами,
чтобы избежать засорения общими/неправильными изображениями.

Запускать из корня проекта:
    python scripts/scrape_all.py
"""
import shutil
from pathlib import Path
from icrawler.builtin import BingImageCrawler

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset" / "raw"

# Цель: ~60 картинок / класс. Для каждого класса набор конкретных запросов
# (избегаем общих слов типа "notebook" — даёт ноутбуки; "wallet" — даёт пустую кожу).
TARGETS = {
    "bag": {
        "needed": 60,
        "queries": [
            "tote bag product photo white background",
            "crossbody handbag advertising",
            "leather shoulder bag isolated",
            "clutch evening bag product",
            "женская сумка реклама",
            "luxury handbag campaign",
        ],
    },
    "chair": {
        "needed": 60,
        "queries": [
            "office chair product photo",
            "dining chair isolated white background",
            "armchair living room product",
            "modern chair furniture advertising",
            "стул кресло реклама",
            "lounge chair product photography",
        ],
    },
    "clock": {
        "needed": 60,
        "queries": [
            "wall clock product photo",
            "table clock isolated",
            "vintage wall clock advertising",
            "kitchen clock product",
            "настенные часы реклама",
            "alarm clock bedside isolated",
        ],
    },
    "cup": {
        "needed": 60,
        "queries": [
            "coffee mug product photo white background",
            "ceramic cup advertising",
            "tea cup with saucer isolated",
            "travel mug thermos product",
            "кружка чашка реклама",
            "espresso cup product photography",
        ],
    },
    "cutlery": {
        "needed": 60,
        "queries": [
            "cutlery set silverware product photo",
            "fork knife spoon isolated white background",
            "stainless steel flatware advertising",
            "столовые приборы реклама",
            "kitchen knife product photo",
            "dinner fork spoon set",
        ],
    },
    "glasses": {
        "needed": 60,
        "queries": [
            "sunglasses product photo white background",
            "eyeglasses optical frames advertising",
            "aviator sunglasses isolated",
            "очки солнцезащитные реклама",
            "reading glasses product",
            "designer sunglasses campaign",
        ],
    },
    "hat": {
        "needed": 60,
        "queries": [
            "baseball cap product photo white background",
            "fedora hat isolated",
            "winter beanie hat advertising",
            "straw hat summer product",
            "шляпа кепка реклама",
            "wide brim hat product photography",
        ],
    },
    "headphones": {
        "needed": 60,
        "queries": [
            "wireless headphones product photo",
            "over ear headphones advertising",
            "earbuds in ear isolated white background",
            "наушники реклама",
            "gaming headset product",
            "bluetooth headphones campaign",
        ],
    },
    "jacket": {
        "needed": 60,
        "queries": [
            "winter jacket product photo white background",
            "leather jacket men advertising",
            "bomber jacket isolated",
            "puffer jacket product photography",
            "куртка мужская женская реклама",
            "denim jacket fashion campaign",
        ],
    },
    "lipstick": {
        "needed": 60,
        "queries": [
            "lipstick tube product photo white background",
            "matte lipstick isolated",
            "красная помада реклама",
            "lipstick swatch advertising",
            "помада макияж продукт",
            "luxury lipstick beauty campaign",
        ],
    },
    "notebook": {
        "needed": 60,
        "queries": [
            "paper notebook stationery product photo",
            "spiral bound notebook isolated white background",
            "школьная тетрадь в клетку",
            "блокнот бумажный обложка",
            "moleskine journal product",
            "leather diary notebook isolated",
        ],
    },
    "perfume": {
        "needed": 60,
        "queries": [
            "perfume bottle product photo white background",
            "luxury fragrance advertising",
            "духи парфюм реклама",
            "eau de parfum bottle isolated",
            "designer perfume campaign",
            "cologne bottle product photography",
        ],
    },
    "phone": {
        "needed": 60,
        "queries": [
            "smartphone product photo white background",
            "iphone advertising campaign",
            "mobile phone isolated",
            "android smartphone product",
            "смартфон телефон реклама",
            "phone in hand product photography",
        ],
    },
    "shirt": {
        "needed": 60,
        "queries": [
            "t-shirt product photo white background isolated",
            "plain white tshirt mockup",
            "polo shirt advertising",
            "футболка реклама",
            "men casual tshirt product",
            "graphic tee fashion campaign",
        ],
    },
    "shoes": {
        "needed": 60,
        "queries": [
            "sneakers product photo white background",
            "running shoes advertising",
            "leather shoes men isolated",
            "обувь кроссовки реклама",
            "high heels women product",
            "boots product photography",
        ],
    },
    "socks": {
        "needed": 60,
        "queries": [
            "socks pair product photo white background",
            "cotton socks isolated",
            "носки реклама",
            "athletic sport socks product",
            "wool socks advertising",
            "colorful socks fashion",
        ],
    },
    "tie": {
        "needed": 60,
        "queries": [
            "silk necktie product photo white background",
            "tie advertising campaign",
            "галстук мужской реклама",
            "luxury tie folded isolated",
            "men necktie product photography",
            "bow tie product close up",
        ],
    },
    "umbrella": {
        "needed": 60,
        "queries": [
            "umbrella open product photo white background",
            "rain umbrella advertising",
            "зонт зонтик реклама",
            "compact folding umbrella isolated",
            "black umbrella product",
            "colorful umbrella fashion",
        ],
    },
    "wallet": {
        "needed": 60,
        "queries": [
            "bifold wallet card slots open",
            "leather wallet with cards product photo",
            "men wallet open showing cards",
            "кошелёк портмоне с отделениями",
            "card holder wallet isolated",
            "женский кошелёк отделения купюры",
        ],
    },
    "watch": {
        "needed": 60,
        "queries": [
            "wristwatch product photo white background",
            "luxury watch advertising",
            "наручные часы реклама",
            "smartwatch isolated product",
            "men watch leather strap product",
            "designer watch campaign close up",
        ],
    },
}


def count_files(p: Path) -> int:
    return sum(1 for x in p.iterdir() if x.is_file()) if p.exists() else 0


def scrape_class(cls: str, config: dict) -> None:
    folder = DATASET_DIR / cls
    folder.mkdir(parents=True, exist_ok=True)
    needed = config["needed"]
    initial = count_files(folder)

    print(f"\n══ {cls}: было {initial}, цель {needed} ══")

    per_query = needed // len(config["queries"]) + 3

    for q_idx, query in enumerate(config["queries"]):
        if count_files(folder) >= needed:
            break
        tmp = folder / f"_tmp_{q_idx}"
        tmp.mkdir(exist_ok=True)
        print(f"  ▸ '{query}'")
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(tmp)},
                feeder_threads=1, parser_threads=2, downloader_threads=4,
                log_level=40,
            )
            crawler.crawl(keyword=query, max_num=per_query * 2, min_size=(400, 400))
        except Exception as e:
            print(f"    ✗ {e}")

        existing = count_files(folder)
        moved = 0
        for f in tmp.iterdir():
            if not f.is_file():
                continue
            if count_files(folder) >= needed:
                break
            ext = f.suffix.lower() or ".jpg"
            idx = existing + moved + 1
            target = folder / f"{cls}_{idx:04d}{ext}"
            while target.exists():
                moved += 1
                idx = existing + moved + 1
                target = folder / f"{cls}_{idx:04d}{ext}"
            shutil.move(str(f), str(target))
            moved += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"    ✓ +{moved}, всего {count_files(folder)}")

    print(f"  ▸ итог: {count_files(folder)}")


if __name__ == "__main__":
    print(f"Качаем {len(TARGETS)} классов в {DATASET_DIR}\n")
    for cls, cfg in TARGETS.items():
        scrape_class(cls, cfg)
    total = sum(count_files(DATASET_DIR / c) for c in TARGETS)
    print(f"\n✅ Готово. Всего: {total} файлов в {len(TARGETS)} папках")
