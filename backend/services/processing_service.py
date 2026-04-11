import logging
import os
from datetime import datetime
from datetime import timezone

from config import YOLO_CONFIDENCE_THRESHOLD
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from ml_models.classifier import classify_creative
from ml_models.ocr_model import extract_text_and_blocks
from ml_models.yolo_detector import detect_objects
from PIL import Image
from services.settings_service import get_setting
from utils.color_utils import classify_colors_by_palette
from utils.color_utils import get_top_colors


logger = logging.getLogger(__name__)


class CreativeNotFoundError(Exception):
    pass


def get_creative_and_analysis(db, creative_id: str) -> tuple[Creative, CreativeAnalysis]:
    """Получает креатив и его анализ из БД, создаёт анализ если нет."""
    creative = db.query(Creative).filter(Creative.creative_id == creative_id).first()
    if not creative:
        raise CreativeNotFoundError(f"Креатив {creative_id} не найден")

    analysis = db.query(CreativeAnalysis).filter(
        CreativeAnalysis.creative_id == creative_id,
    ).first()

    if not analysis:
        analysis = CreativeAnalysis(creative_id=creative_id)
        db.add(analysis)
        db.commit()
        db.refresh(analysis)

    return creative, analysis


def get_image_dimensions(creative: Creative, image_path: str, db):
    """Извлекает размеры изображения и обновляет запись."""
    try:
        with Image.open(image_path) as img:
            creative.image_width, creative.image_height = img.size
            db.commit()
    except Exception:
        logger.exception("Не удалось получить размеры для %s", creative.creative_id)
        if os.path.exists(image_path):
            os.unlink(image_path)


def perform_ocr(creative: Creative, analysis: CreativeAnalysis, image_path: str, db):
    """Выполняет OCR на изображении."""
    logger.info("OCR для %s", creative.creative_id)
    analysis.ocr_status = "PROCESSING"
    start_at = datetime.now(timezone.utc)
    analysis.ocr_start = start_at
    db.commit()

    try:
        text, blocks = extract_text_and_blocks(image_path, creative)
        analysis.ocr_text = text
        analysis.ocr_blocks = blocks
        analysis.ocr_status = "SUCCESS"
    except Exception:
        logger.exception("Ошибка OCR для %s", creative.creative_id)
        analysis.ocr_status = "ERROR"

    end_at = datetime.now(timezone.utc)
    analysis.ocr_end = end_at
    analysis.ocr_duration = (end_at - start_at).total_seconds()
    db.commit()


def perform_detection(creative: Creative, analysis: CreativeAnalysis, image_path: str, db):
    """Выполняет детекцию объектов через YOLO."""
    logger.info("Детекция для %s", creative.creative_id)
    analysis.detection_status = "PROCESSING"
    start_at = datetime.now(timezone.utc)
    analysis.detection_start = start_at
    db.commit()

    try:
        objects = detect_objects(image_path, confidence=YOLO_CONFIDENCE_THRESHOLD)
        analysis.detected_objects = objects
        analysis.detection_status = "SUCCESS"
    except Exception:
        logger.exception("Ошибка детекции для %s", creative.creative_id)
        analysis.detection_status = "ERROR"

    end_at = datetime.now(timezone.utc)
    analysis.detection_end = end_at
    analysis.detection_duration = (end_at - start_at).total_seconds()
    db.commit()


def perform_classification(
    creative: Creative,
    analysis: CreativeAnalysis,
    db,
    image_path: str | None = None,
):
    """Классифицирует креатив по тематике на основе OCR, детекции и CLIP-фичей изображения."""
    logger.info("Классификация для %s", creative.creative_id)
    analysis.classification_status = "PROCESSING"
    start_at = datetime.now(timezone.utc)
    analysis.classification_start = start_at
    db.commit()

    try:
        topic, confidence = classify_creative(
            ocr_text=analysis.ocr_text,
            detected_objects=analysis.detected_objects,
            image_path=image_path,
        )
        analysis.main_topic = topic
        analysis.topic_confidence = confidence
        analysis.classification_status = "SUCCESS"
    except Exception:
        logger.exception("Ошибка классификации для %s", creative.creative_id)
        analysis.classification_status = "ERROR"

    end_at = datetime.now(timezone.utc)
    analysis.classification_end = end_at
    analysis.classification_duration = (end_at - start_at).total_seconds()
    db.commit()


def perform_color_analysis(creative: Creative, analysis: CreativeAnalysis, image_path: str, db):
    """Анализирует цвета изображения."""
    logger.info("Анализ цветов для %s", creative.creative_id)
    analysis.color_status = "PROCESSING"
    start_at = datetime.now(timezone.utc)
    analysis.color_start = start_at
    db.commit()

    try:
        n_dominant = get_setting(db, "DOMINANT_COLORS_COUNT", 3)
        n_secondary = get_setting(db, "SECONDARY_COLORS_COUNT", 3)

        dominant = get_top_colors(image_path, n_colors=n_dominant)
        secondary = get_top_colors(image_path, n_colors=n_dominant + n_secondary)
        secondary = secondary[n_dominant:]

        palette = classify_colors_by_palette(dominant + secondary)

        analysis.dominant_colors = dominant
        analysis.secondary_colors = secondary
        analysis.palette_colors = palette
        analysis.color_status = "SUCCESS"
    except Exception:
        logger.exception("Ошибка анализа цветов для %s", creative.creative_id)
        analysis.color_status = "ERROR"

    end_at = datetime.now(timezone.utc)
    analysis.color_end = end_at
    analysis.color_duration = (end_at - start_at).total_seconds()
    db.commit()
