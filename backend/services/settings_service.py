import logging

from database_models.app_settings import AppSettings
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)


def get_setting(db: Session, key: str, default=None):
    setting = db.query(AppSettings).filter(AppSettings.key == key).first()
    if setting:
        return setting.get_value()
    return default


def update_setting(db: Session, key: str, value):
    setting = db.query(AppSettings).filter(AppSettings.key == key).first()
    if setting:
        old_value = setting.value
        setting.set_value(value)
        db.commit()
        db.refresh(setting)
        logger.info("Настройка %s обновлена: %s → %s", key, old_value, setting.value)
        return setting

    logger.warning("Настройка %s не найдена", key)
    return None


def get_all_settings(db: Session) -> dict:
    settings = db.query(AppSettings).all()
    return {s.key: s.get_value() for s in settings}
