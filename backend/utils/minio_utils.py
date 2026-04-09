import logging
import os
from pathlib import Path

from config import settings
from minio.error import S3Error
from minio_client import minio_client


logger = logging.getLogger(__name__)


class FileNotSavedException(Exception):
    def __init__(self, path: str):
        super().__init__(f"Файл не сохранён локально: {path}")


def upload_to_minio(local_path: str, object_name: str) -> str:
    """Загружает файл в MinIO. Возвращает путь объекта."""
    bucket = settings.MINIO_BUCKET

    if not minio_client.bucket_exists(bucket):
        minio_client.make_bucket(bucket)

    try:
        minio_client.fput_object(bucket, object_name, local_path)
        logger.info("Загружен в MinIO: %s/%s", bucket, object_name)
        return object_name
    except (S3Error, Exception):
        logger.exception("Ошибка загрузки в MinIO: %s", object_name)
        raise


def download_file_from_minio(creative, analysis, db, local_path: str):
    """Скачивает файл креатива из MinIO в локальный путь."""
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        minio_client.fget_object(settings.MINIO_BUCKET, creative.file_path, local_path)

        if not os.path.exists(local_path):
            raise FileNotSavedException(local_path)

        logger.info("Скачан из MinIO: %s → %s", creative.file_path, local_path)

    except S3Error:
        logger.exception("S3 ошибка при скачивании %s", creative.file_path)
        analysis.overall_status = "ERROR"
        analysis.error_message = f"Не удалось скачать из MinIO: {creative.file_path}"
        db.commit()
        _cleanup_file(local_path)

    except Exception:
        logger.exception("Ошибка при скачивании %s", creative.file_path)
        analysis.overall_status = "ERROR"
        analysis.error_message = f"Ошибка скачивания: {creative.file_path}"
        db.commit()
        _cleanup_file(local_path)


def _cleanup_file(path: str):
    """Удаляет файл если существует."""
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        logger.warning("Не удалось удалить %s", path)
