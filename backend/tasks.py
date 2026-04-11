import logging
import os
import tempfile
from datetime import datetime
from datetime import timezone

from celery import Celery
from config import settings
from database import SessionLocal
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from services.processing_service import get_creative_and_analysis
from services.processing_service import get_image_dimensions
from services.processing_service import perform_classification
from services.processing_service import perform_color_analysis
from services.processing_service import perform_detection
from services.processing_service import perform_ocr
from utils.minio_utils import download_file_from_minio


logger = logging.getLogger(__name__)

celery = Celery("tasks", broker=settings.REDIS_URL, backend=settings.REDIS_URL)


@celery.task(bind=True, max_retries=3)
def process_creative(self, creative_id: str):
    """Основная задача обработки креатива через ML-пайплайн."""
    logger.info("Начало обработки креатива %s", creative_id)

    db = SessionLocal()
    tmp_path = None

    try:
        creative, analysis = get_creative_and_analysis(db, creative_id)

        analysis.overall_status = "PROCESSING"
        processing_start_at = datetime.now(timezone.utc)
        analysis.processing_start = processing_start_at
        db.commit()

        # Скачиваем изображение из MinIO
        suffix = f".{creative.file_format}" if creative.file_format else ".jpg"
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_path = tmp_file.name
        tmp_file.close()

        download_file_from_minio(creative, analysis, db, tmp_path)

        if analysis.overall_status == "ERROR":
            return

        # Получаем размеры изображения
        get_image_dimensions(creative, tmp_path, db)

        # ML-пайплайн
        perform_ocr(creative, analysis, tmp_path, db)
        perform_detection(creative, analysis, tmp_path, db)
        perform_classification(creative, analysis, db, image_path=tmp_path)
        perform_color_analysis(creative, analysis, tmp_path, db)

        # Завершение
        analysis.overall_status = "SUCCESS"
        processing_end_at = datetime.now(timezone.utc)
        analysis.processing_end = processing_end_at
        analysis.total_duration = (
            processing_end_at - processing_start_at
        ).total_seconds()
        db.commit()

        logger.info(
            "Креатив %s обработан за %.2f сек",
            creative_id,
            analysis.total_duration,
        )

    except Exception as exc:
        db.rollback()
        logger.exception("Ошибка при обработке креатива %s", creative_id)

        try:
            if analysis:
                analysis.overall_status = "ERROR"
                analysis.error_message = str(exc)
                analysis.processing_end = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            logger.exception("Не удалось обновить статус ошибки для %s", creative_id)

        raise self.retry(exc=exc, countdown=5)

    finally:
        db.close()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Не удалось удалить временный файл %s", tmp_path)
