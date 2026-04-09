import logging
from contextlib import asynccontextmanager

from config import DOMINANT_COLORS_COUNT
from config import SECONDARY_COLORS_COUNT
from config import settings
from database import Base
from database import SessionLocal
from database import engine
from database_models.app_settings import AppSettings
from minio_client import minio_client


logger = logging.getLogger(__name__)


def initialize_default_settings(db):
    """Создаёт настройки по умолчанию, если их нет в БД."""
    defaults = {
        "DOMINANT_COLORS_COUNT": (str(DOMINANT_COLORS_COUNT), "Количество доминирующих цветов"),
        "SECONDARY_COLORS_COUNT": (str(SECONDARY_COLORS_COUNT), "Количество второстепенных цветов"),
    }

    for key, (value, description) in defaults.items():
        existing = db.query(AppSettings).filter(AppSettings.key == key).first()
        if not existing:
            setting = AppSettings(key=key, value=value, description=description)
            db.add(setting)
            logger.info("Создана настройка по умолчанию: %s = %s", key, value)

    db.commit()


@asynccontextmanager
async def lifespan(app):
    """Инициализация при запуске и очистка при остановке."""
    logger.info("Запуск приложения...")

    # Создаём таблицы
    Base.metadata.create_all(bind=engine)
    logger.info("Таблицы БД созданы")

    # Настройки по умолчанию
    db = SessionLocal()
    try:
        initialize_default_settings(db)
    finally:
        db.close()

    # Проверяем MinIO-бакет
    bucket = settings.MINIO_BUCKET
    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)
        logger.info("Создан бакет MinIO: %s", bucket)

    yield

    logger.info("Приложение остановлено")
