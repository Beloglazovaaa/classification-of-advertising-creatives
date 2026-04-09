import logging
from datetime import datetime
from datetime import timezone

from config import ML_STAGES
from database import get_db
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)
router = APIRouter()


def format_status_with_time(status: str | None, start: datetime | None, duration: float | None) -> str:
    """Форматирует статус стадии с временем выполнения."""
    if not status or status == "PENDING":
        return "—"
    if status == "ERROR":
        return "X"
    if status == "SUCCESS" and duration is not None:
        return f"{duration:.1f}sec"
    if status == "PROCESSING" and start:
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        return f"{elapsed:.1f}sec "
    return "—"


@router.get("/status/{creative_id}")
def get_creative_status(creative_id: str, db: Session = Depends(get_db)):
    """Возвращает статус обработки креатива."""
    creative = db.query(Creative).filter(Creative.creative_id == creative_id).first()
    if not creative:
        raise HTTPException(status_code=404, detail="Креатив не найден")

    analysis = db.query(CreativeAnalysis).filter(
        CreativeAnalysis.creative_id == creative_id,
    ).first()

    result = {
        "creative_id": creative.creative_id,
        "original_filename": creative.original_filename,
        "file_size": creative.file_size,
        "image_size": f"{creative.image_width}x{creative.image_height}" if creative.image_width else "—",
        "upload_timestamp": creative.upload_timestamp.isoformat() if creative.upload_timestamp else None,
        "main_topic": None,
        "topic_confidence": None,
        "overall_status": "PENDING",
    }

    if analysis:
        result["main_topic"] = analysis.main_topic
        result["topic_confidence"] = analysis.topic_confidence
        result["overall_status"] = analysis.overall_status

        for stage in ML_STAGES:
            status_val = getattr(analysis, stage["status_field"], None)
            start_val = getattr(analysis, stage["start_field"], None)
            duration_val = getattr(analysis, stage["duration_field"], None)
            result[stage["status_field"]] = format_status_with_time(status_val, start_val, duration_val)
    else:
        for stage in ML_STAGES:
            result[stage["status_field"]] = "—"

    return result
