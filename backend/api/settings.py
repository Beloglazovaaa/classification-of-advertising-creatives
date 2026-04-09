import logging

from database import get_db
from database_models.app_settings import AppSettings
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from services.settings_service import get_all_settings
from services.settings_service import update_setting
from sqlalchemy.orm import Session


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings")


@router.get("/")
def get_settings(db: Session = Depends(get_db)):
    """Возвращает все настройки приложения."""
    return get_all_settings(db)


@router.get("/{key}")
def get_setting_by_key(key: str, db: Session = Depends(get_db)):
    """Возвращает конкретную настройку."""
    setting = db.query(AppSettings).filter(AppSettings.key == key).first()
    if not setting:
        raise HTTPException(status_code=404, detail=f"Настройка '{key}' не найдена")
    return {"key": setting.key, "value": setting.get_value(), "description": setting.description}


@router.put("/{key}")
def update_single_setting(key: str, body: dict, db: Session = Depends(get_db)):
    """Обновляет одну настройку."""
    value = body.get("value")
    if value is None:
        raise HTTPException(status_code=400, detail="Не указано значение")

    result = update_setting(db, key, value)
    if not result:
        raise HTTPException(status_code=404, detail=f"Настройка '{key}' не найдена")

    return {"key": result.key, "value": result.get_value()}


@router.put("/")
def update_settings_batch(updates: dict, db: Session = Depends(get_db)):
    """Обновляет несколько настроек одновременно."""
    results = {}
    for key, value in updates.items():
        result = update_setting(db, key, value)
        if result:
            results[key] = result.get_value()
        else:
            logger.warning("Настройка %s не найдена", key)

    return get_all_settings(db)
