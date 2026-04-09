import logging
from collections import defaultdict

from config import PRODUCT_TOPICS
from database import get_db
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from fastapi import APIRouter
from fastapi import Depends
from models import AnalysisSummary
from models import AnalyticsResponse
from models import TopicItem
from services.analytics_service import calculate_group_processing_time
from services.analytics_service import get_color_class_distribution
from services.analytics_service import get_topic_color_distribution
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)
router = APIRouter()


def _build_analytics(analyses: list[CreativeAnalysis], db: Session, group_id: str | None) -> dict:
    """Общая логика построения аналитики (для группы или для всех)."""
    if not analyses:
        return AnalyticsResponse(summary=AnalysisSummary()).model_dump()

    total = len(analyses)
    ocr_confs = []
    obj_confs = []
    topic_confs = []
    topic_counts = defaultdict(int)

    for a in analyses:
        # OCR confidence
        if a.ocr_blocks:
            for block in a.ocr_blocks:
                conf = block.get("confidence", 0)
                if conf > 0:
                    ocr_confs.append(conf)

        # Object detection confidence
        if a.detected_objects:
            for obj in a.detected_objects:
                conf = obj.get("confidence", 0)
                if conf > 0:
                    obj_confs.append(conf)

        # Topic confidence
        if a.topic_confidence and a.topic_confidence > 0:
            topic_confs.append(a.topic_confidence)

        if a.main_topic:
            topic_counts[a.main_topic] += 1

    summary = AnalysisSummary(
        total_creatives=total,
        avg_ocr_confidence=sum(ocr_confs) / len(ocr_confs) if ocr_confs else 0.0,
        avg_object_confidence=sum(obj_confs) / len(obj_confs) if obj_confs else 0.0,
        avg_topic_confidence=sum(topic_confs) / len(topic_confs) if topic_confs else 0.0,
    )

    topics = [TopicItem(topic=t, count=c) for t, c in topic_counts.items()]
    topics.sort(key=lambda x: x.count, reverse=True)

    # Таблица по топикам
    topics_table = []
    for topic, count in topic_counts.items():
        topic_analyses = [a for a in analyses if a.main_topic == topic]
        topic_ocr = []
        topic_obj = []
        topic_topic = []

        for a in topic_analyses:
            if a.ocr_blocks:
                topic_ocr.extend(b.get("confidence", 0) for b in a.ocr_blocks if b.get("confidence", 0) > 0)
            if a.detected_objects:
                topic_obj.extend(o.get("confidence", 0) for o in a.detected_objects if o.get("confidence", 0) > 0)
            if a.topic_confidence and a.topic_confidence > 0:
                topic_topic.append(a.topic_confidence)

        ru_name = PRODUCT_TOPICS.get(topic, {}).get("ru", topic)
        topics_table.append({
            "Тематика": ru_name,
            "Количество": count,
            "Ср. OCR conf": round(sum(topic_ocr) / len(topic_ocr), 3) if topic_ocr else 0,
            "Ср. Object conf": round(sum(topic_obj) / len(topic_obj), 3) if topic_obj else 0,
            "Ср. Topic conf": round(sum(topic_topic) / len(topic_topic), 3) if topic_topic else 0,
        })

    # Цвета
    color_class_dist = get_color_class_distribution(analyses)
    topic_color_dist = get_topic_color_distribution(analyses)

    processing_time = calculate_group_processing_time(db, group_id)

    return AnalyticsResponse(
        summary=summary,
        topics=[t.model_dump() for t in topics],
        color_class_distribution=color_class_dist,
        topic_color_distribution=topic_color_dist,
        topics_table=topics_table,
        total_processing_time=processing_time,
        total_creatives_in_group=total,
    ).model_dump()


@router.get("/analytics/group/{group_id}")
def get_group_analytics(group_id: str, db: Session = Depends(get_db)):
    """Аналитика по конкретной группе."""
    creative_ids = [
        cid for (cid,) in db.query(Creative.creative_id).filter(Creative.group_id == group_id).all()
    ]

    analyses = db.query(CreativeAnalysis).filter(
        CreativeAnalysis.creative_id.in_(creative_ids),
        CreativeAnalysis.overall_status == "SUCCESS",
    ).all()

    return _build_analytics(analyses, db, group_id)


@router.get("/analytics/all")
def get_all_analytics(db: Session = Depends(get_db)):
    """Аналитика по всем креативам."""
    analyses = db.query(CreativeAnalysis).filter(
        CreativeAnalysis.overall_status == "SUCCESS",
    ).all()

    return _build_analytics(analyses, db, group_id=None)
