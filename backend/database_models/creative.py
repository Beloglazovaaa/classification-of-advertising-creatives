from datetime import datetime
from datetime import timezone

from database import Base
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.dialects.postgresql import JSON


class Creative(Base):
    __tablename__ = "creatives"

    creative_id = Column(String, primary_key=True, index=True)
    group_id = Column(String, index=True, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    file_format = Column(String)
    image_width = Column(Integer)
    image_height = Column(Integer)
    upload_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class CreativeAnalysis(Base):
    __tablename__ = "creative_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    creative_id = Column(String, ForeignKey("creatives.creative_id"), index=True, nullable=False)

    # Результаты OCR
    ocr_text = Column(Text, nullable=True)
    ocr_blocks = Column(JSON, nullable=True)

    # Результаты детекции объектов
    detected_objects = Column(JSON, nullable=True)

    # Результаты классификации
    main_topic = Column(String, nullable=True)
    topic_confidence = Column(Float, nullable=True)

    # Результаты анализа цветов
    dominant_colors = Column(JSON, nullable=True)
    secondary_colors = Column(JSON, nullable=True)
    palette_colors = Column(JSON, nullable=True)

    # Статусы стадий
    ocr_status = Column(String, default="PENDING")
    detection_status = Column(String, default="PENDING")
    classification_status = Column(String, default="PENDING")
    color_status = Column(String, default="PENDING")
    overall_status = Column(String, default="PENDING")

    # Таймстемпы стадий
    ocr_start = Column(DateTime, nullable=True)
    ocr_end = Column(DateTime, nullable=True)
    ocr_duration = Column(Float, nullable=True)

    detection_start = Column(DateTime, nullable=True)
    detection_end = Column(DateTime, nullable=True)
    detection_duration = Column(Float, nullable=True)

    classification_start = Column(DateTime, nullable=True)
    classification_end = Column(DateTime, nullable=True)
    classification_duration = Column(Float, nullable=True)

    color_start = Column(DateTime, nullable=True)
    color_end = Column(DateTime, nullable=True)
    color_duration = Column(Float, nullable=True)

    # Общие метрики
    processing_start = Column(DateTime, nullable=True)
    processing_end = Column(DateTime, nullable=True)
    total_duration = Column(Float, nullable=True)

    error_message = Column(Text, nullable=True)
