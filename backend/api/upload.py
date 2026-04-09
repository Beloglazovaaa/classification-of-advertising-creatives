import logging
import os
from pathlib import Path

from config import settings
from database import get_db
from fastapi import APIRouter
from fastapi import Depends
from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from models import UploadResponse
from PIL import Image
from services.upload_service import create_creative
from sqlalchemy.orm import Session
from tasks import process_creative
from utils.minio_utils import upload_to_minio


logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


@router.post("/upload", response_model=UploadResponse)
def upload_creatives(
    files: list[UploadFile] = File(...),
    group_id: str = Form(...),
    creative_ids: list[str] = Form(...),
    original_filenames: list[str] = Form(...),
    db: Session = Depends(get_db),
):
    """Загружает креативы, сохраняет в MinIO и ставит в очередь на обработку."""
    if len(files) != len(creative_ids) or len(files) != len(original_filenames):
        return UploadResponse(
            uploaded=0,
            group_id=group_id,
            errors=["Количество файлов, ID и имён должно совпадать"],
        )

    uploaded_count = 0
    errors = []
    upload_dir = Path(settings.UPLOAD_FOLDER)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file, creative_id, original_name in zip(files, creative_ids, original_filenames):
        try:
            ext = file.filename.split(".")[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"Неподдерживаемый формат: {file.filename}")
                continue

            # Сохраняем временно
            filename = f"{creative_id}.{ext}"
            tmp_path = upload_dir / filename

            with open(tmp_path, "wb") as f:
                content = file.file.read()
                f.write(content)

            # Размеры изображения
            with Image.open(tmp_path) as img:
                width, height = img.size

            # Загружаем в MinIO
            object_name = f"{group_id}/{filename}"
            minio_path = upload_to_minio(str(tmp_path), object_name)

            # Сохраняем в БД
            create_creative(
                db=db,
                creative_id=creative_id,
                group_id=group_id,
                original_filename=original_name,
                minio_path=minio_path,
                file_size=len(content),
                file_format=ext,
                image_width=width,
                image_height=height,
            )

            # Ставим в очередь на обработку
            process_creative.delay(creative_id)
            uploaded_count += 1

            logger.info("Загружен креатив %s (%s)", creative_id, original_name)

        except Exception as exc:
            logger.exception("Ошибка загрузки файла %s", file.filename)
            errors.append(f"Ошибка для {file.filename}: {exc}")

        finally:
            if tmp_path.exists():
                os.unlink(tmp_path)

    return UploadResponse(uploaded=uploaded_count, group_id=group_id, errors=errors)
