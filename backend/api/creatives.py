import logging

from config import settings
from database import get_db
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)
router = APIRouter()


def _build_public_url(file_path: str) -> str:
    """Формирует публичный URL для файла в MinIO."""
    return f"{settings.MINIO_PUBLIC_URL}/{settings.MINIO_BUCKET}/{file_path}"


@router.get("/creatives/{creative_id}")
def get_creative_details(creative_id: str, db: Session = Depends(get_db)):
    """Возвращает детали креатива с результатами анализа."""
    creative = db.query(Creative).filter(Creative.creative_id == creative_id).first()
    if not creative:
        raise HTTPException(status_code=404, detail="Креатив не найден")

    analysis = db.query(CreativeAnalysis).filter(
        CreativeAnalysis.creative_id == creative_id,
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Анализ не найден")

    result = {
        "creative_id": creative.creative_id,
        "group_id": creative.group_id,
        "original_filename": creative.original_filename,
        "file_path": _build_public_url(creative.file_path),
        "file_size": creative.file_size,
        "file_format": creative.file_format,
        "image_width": creative.image_width,
        "image_height": creative.image_height,
        "upload_timestamp": creative.upload_timestamp.isoformat() if creative.upload_timestamp else None,
        "overall_status": analysis.overall_status,
    }

    if analysis.overall_status == "ERROR":
        raise HTTPException(status_code=500, detail="Ошибка обработки креатива")

    if analysis.overall_status == "SUCCESS":
        result.update({
            "ocr_text": analysis.ocr_text,
            "ocr_blocks": analysis.ocr_blocks,
            "detected_objects": analysis.detected_objects,
            "main_topic": analysis.main_topic,
            "topic_confidence": analysis.topic_confidence,
            "dominant_colors": analysis.dominant_colors,
            "secondary_colors": analysis.secondary_colors,
            "palette_colors": analysis.palette_colors,
        })

    return result


@router.get("/groups/{group_id}/creatives")
def get_creatives_by_group(group_id: str, db: Session = Depends(get_db)):
    """Возвращает список креативов в группе."""
    creatives = db.query(Creative).filter(Creative.group_id == group_id).all()

    result = []
    for creative in creatives:
        analysis = db.query(CreativeAnalysis).filter(
            CreativeAnalysis.creative_id == creative.creative_id,
            CreativeAnalysis.overall_status == "SUCCESS",
        ).first()

        result.append({
            "creative_id": creative.creative_id,
            "original_filename": creative.original_filename,
            "file_format": creative.file_format,
            "image_width": creative.image_width,
            "image_height": creative.image_height,
            "upload_timestamp": creative.upload_timestamp.isoformat() if creative.upload_timestamp else None,
            "analysis": analysis is not None,
        })

    return result
