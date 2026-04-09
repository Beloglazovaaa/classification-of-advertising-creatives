import logging

from database import get_db
from database_models.creative import Creative
from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy import func
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/groups")
def get_groups(db: Session = Depends(get_db)):
    """Возвращает список всех групп креативов."""
    group_ids = db.query(Creative.group_id).distinct().all()

    result = []
    for (group_id,) in group_ids:
        count = db.query(func.count(Creative.creative_id)).filter(
            Creative.group_id == group_id,
        ).scalar()

        earliest = db.query(func.min(Creative.upload_timestamp)).filter(
            Creative.group_id == group_id,
        ).scalar()

        result.append({
            "group_id": group_id,
            "total_creatives": count,
            "created_at": earliest.isoformat() if earliest else None,
        })

    result.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return result
