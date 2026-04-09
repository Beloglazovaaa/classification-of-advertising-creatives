import logging
from collections import defaultdict

from config import COLOR_VISUAL_HEX
from database_models.creative import Creative
from database_models.creative import CreativeAnalysis
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)

TOP_COLORS_PER_TOPIC = 5


def calculate_group_processing_time(db: Session, group_id: str | None) -> float:
    """Считает общее время обработки для группы (или всех)."""
    query = db.query(CreativeAnalysis)

    if group_id:
        creative_ids = [
            cid for (cid,) in db.query(Creative.creative_id).filter(Creative.group_id == group_id).all()
        ]
        query = query.filter(CreativeAnalysis.creative_id.in_(creative_ids))

    analyses = query.filter(CreativeAnalysis.overall_status == "SUCCESS").all()

    total = 0.0
    for a in analyses:
        if a.total_duration:
            total += a.total_duration

    return total


def get_color_class_distribution(analyses: list[CreativeAnalysis]) -> dict[str, float]:
    """Агрегирует распределение цветов по палитре из всех анализов."""
    color_totals = defaultdict(float)

    for a in analyses:
        if not a.palette_colors:
            continue
        for cls_name, info in a.palette_colors.items():
            percent = info.get("percent", 0) if isinstance(info, dict) else 0
            color_totals[cls_name] += percent

    # Нормализуем
    total = sum(color_totals.values())
    if total > 0:
        return {k: round(v / total * 100, 2) for k, v in color_totals.items()}
    return {}


def get_topic_color_distribution(analyses: list[CreativeAnalysis], top_n: int = TOP_COLORS_PER_TOPIC) -> dict:
    """Строит распределение цветов по топикам."""
    topic_colors = defaultdict(lambda: defaultdict(float))

    for a in analyses:
        if not a.main_topic or not a.palette_colors:
            continue
        for cls_name, info in a.palette_colors.items():
            percent = info.get("percent", 0) if isinstance(info, dict) else 0
            topic_colors[a.main_topic][cls_name] += percent

    result = {}
    for topic, colors in topic_colors.items():
        sorted_colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Нормализуем топ-N к 100%
        total = sum(p for _, p in sorted_colors)
        if total > 0:
            result[topic] = [
                {
                    "class": cls_name,
                    "percent": round(pct / total * 100, 1),
                    "hex": COLOR_VISUAL_HEX.get(cls_name, "#ffffff"),
                }
                for cls_name, pct in sorted_colors
            ]
        else:
            result[topic] = []

    return result
