import logging

from database_models.creative import Creative


logger = logging.getLogger(__name__)


def create_creative(
    db,
    creative_id: str,
    group_id: str,
    original_filename: str,
    minio_path: str,
    file_size: int,
    file_format: str,
    image_width: int,
    image_height: int,
):
    """Создаёт и сохраняет запись о креативе в БД."""
    try:
        creative = Creative(
            creative_id=creative_id,
            group_id=group_id,
            original_filename=original_filename,
            file_path=minio_path,
            file_size=file_size,
            file_format=file_format,
            image_width=image_width,
            image_height=image_height,
        )
        db.add(creative)
        db.commit()
        db.refresh(creative)
    except Exception:
        db.rollback()
        logger.exception("Ошибка при сохранении креатива %s в БД", creative_id)
        raise
    else:
        return creative
