# ✨ AdCreativeAI — Классификатор рекламных креативов

ML-платформа для автоматического анализа рекламных баннеров: OCR, детекция объектов, классификация по топикам и извлечение цветовой палитры.

> Улучшенная авторская версия проекта, вдохновлённого [OrlovAlexandr/ad_creatives_classification_mvp](https://github.com/OrlovAlexandr/ad_creatives_classification_mvp).

---

## 🚀 Возможности

- **OCR** — распознавание текста на баннере (EasyOCR, ru + en)
- **Детекция объектов** — YOLOv8m (80 COCO-классов)
- **Классификация топика** — мультимодальный BERT (текст + YOLO-вектор) → 5 категорий
- **Цветовая аналитика** — KMeans в HSV с 17 цветовыми классами
- **Асинхронный пайплайн** — Celery + Redis для параллельной обработки
- **Аналитика** — распределение топиков, цветов, статистика по группам
- **Graceful degradation** — сервис работает даже без обученных моделей (fallback на COCO→топик)

## 🧩 Топики

`bags` · `clocks` · `cups` · `cutlery` · `ties`

## 🏗 Архитектура

```
┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Streamlit│───▶│ FastAPI  │───▶│ Celery worker│
│ frontend │    │ backend  │    │ (ML pipeline)│
└──────────┘    └────┬─────┘    └──────┬───────┘
                     │                 │
              ┌──────┴──────┐   ┌──────┴──────┐
              │ PostgreSQL  │   │    MinIO    │
              │  (metadata) │   │  (images)   │
              └─────────────┘   └─────────────┘
                     ▲
                     │
                  ┌──┴───┐
                  │Redis │
                  └──────┘
```

## ⚡ Быстрый старт

### 1. Клонирование

```bash
git clone https://github.com/Beloglazovaaa/classification-of-advertising-creatives.git
cd classification-of-advertising-creatives
```

### 2. Настройка `.env`

Создайте `.env` в корне проекта:

```env
POSTGRES_USER=creatives
POSTGRES_PASSWORD=creatives
POSTGRES_DB=creatives
DATABASE_URL=postgresql+psycopg2://creatives:creatives@db:5432/creatives

REDIS_URL=redis://redis:6379/0

MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=creatives
MODEL_MINIO_BUCKET=models
MINIO_SECURE=false

BACKEND_URL=http://backend:8000
CELERY_CONCURRENCY=2
```

### 3. Запуск

```bash
docker-compose up --build
```

- Frontend: http://localhost:8501
- Backend API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001 (minioadmin / minioadmin)

## 🧠 Модели

Сервис использует три модели, которые лежат в `minio_init/models/` и автоматически
заливаются в MinIO при первом запуске контейнера `minio_init`:

```
minio_init/models/
├── yolov8m.pt                  ← YOLOv8m (52 MB)
├── best_multimodal_bert.pt     ← Multimodal BERT (681 MB)
└── easy_ocr/
    ├── craft_mlt_25k.pth       ← Text detector (79 MB)
    ├── cyrillic_g2.pth         ← RU recognizer (15 MB)
    └── english_g2.pth          ← EN recognizer (14 MB)
```

> ⚠️ Файлы `.pt` / `.pth` исключены из git через `.gitignore` (слишком тяжёлые).
> Скопируй их вручную или собери через DVC / git-lfs / S3.

### Вариант A: Работа без обученных моделей (demo)

Если моделей нет — **сервис всё равно запустится**. Пайплайн деградирует:

- `OCR` → возвращает пустой текст
- `YOLO` → возвращает пустой список детекций
- `BERT` → fallback на маппинг COCO-классов в топики (`tie` → `ties`, `handbag` → `bags`, и т. д.)

Это позволяет протестировать весь UI и API, но точность классификации будет низкой.

### Вариант B: Скачать готовые веса

```bash
# YOLOv8m
mkdir -p models
curl -L -o models/yolov8m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt

# EasyOCR — запустите сервис один раз с download_enabled=True,
# либо скачайте вручную с https://www.jaided.ai/easyocr/modelhub/
```

Положите веса в `minio_init/models/`, при первом запуске `minio_init` контейнер зальёт их в MinIO.

### Вариант C: Обучить / дообучить BERT-классификатор

Архитектура (`backend/ml_models/classifier.py`):

```
BERT (bert-base-multilingual-cased) ─┐
                                     ├─▶ concat ─▶ FC(256) ─▶ Dropout(0.3) ─▶ FC(5 topics)
YOLO-вектор (80 COCO classes, conf) ─┘
```

Базовая метрика проекта: **accuracy ≈ 0.8641** (CPU).

Готовый training-скрипт: `backend/ml_models/train_bert.py`

**Формат CSV:**
```csv
image_path,ocr_text,yolo_classes,yolo_confs,topic
img_001.jpg,"sale 50% off","tie;handbag","0.91;0.74",ties
img_002.jpg,"summer cups","cup;bowl","0.88;0.65",cups
```

**Запуск (с нуля):**
```bash
cd backend
python -m ml_models.train_bert \
    --csv data/train.csv \
    --output ../minio_init/models/best_multimodal_bert.pt \
    --epochs 5 --batch-size 16 --lr 2e-5
```

**Дообучение поверх существующего чекпоинта:**
```bash
python -m ml_models.train_bert \
    --csv data/new_train.csv \
    --output ../minio_init/models/best_multimodal_bert.pt \
    --resume ../minio_init/models/best_multimodal_bert.pt \
    --epochs 3 --freeze-bert
```

Флаг `--freeze-bert` замораживает энкодер и обучает только классификационную голову —
полезно для маленьких датасетов и быстрых итераций.

После обучения перезапусти `docker-compose up -d --build minio_init backend celery_worker`,
чтобы новые веса попали в MinIO и подтянулись воркером.

## 🎨 Цветовой анализ (CIE Lab)

Перцептивно равномерное пространство **CIE Lab** + **Delta E (CIE76)** для классификации.
Это намного точнее, чем HSV, особенно на смешанных тонах:

- KMeans кластеризует пиксели в Lab → центры в Lab.
- Каждый центр маппится на ближайший цвет из палитры (22 опорных цвета) по евклидову
  расстоянию в Lab.
- Тёмные пиксели (`L < 18`) → "Чёрный", низкая хроматичность → ахроматический ряд.

См. `backend/utils/color_utils.py`.

## 🎨 Frontend

Кастомная тёмная тема (`frontend/components/theme.py`) с градиентами, стеклянными карточками и hero-блоками. Темы страниц:

- 📤 **Загрузка** — drag & drop + live-статус пайплайна с метриками
- 📊 **Аналитика** — Plotly-дашборды с табами (сводка / топики / цвета / таблицы)
- 🔍 **Детали** — OCR-блоки, bounding boxes, палитра
- ⚙️ **Настройки** — конфиг цветового анализа

## 🛠 Стек

- **Backend**: FastAPI, Celery, SQLAlchemy, Pydantic-settings, MinIO
- **ML**: PyTorch, transformers, ultralytics, easyocr, scikit-learn, opencv
- **Frontend**: Streamlit, Plotly, pandas
- **Infra**: Docker Compose, PostgreSQL 15, Redis 7, MinIO

## 🧪 Линтинг

```bash
ruff check backend frontend
```

Настройки в `pyproject.toml` (line-length 120, строгие правила `RUF/E/W/F/N/I/UP/B/...`).

## 📜 Лицензия

MIT · Made with 💜 by [Beloglazovaaa](https://github.com/Beloglazovaaa)
