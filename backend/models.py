from pydantic import BaseModel


class UploadResponse(BaseModel):
    uploaded: int
    group_id: str
    errors: list[str] = []


class SettingUpdate(BaseModel):
    key: str
    value: str | int | float | bool


class SettingResponse(BaseModel):
    key: str
    value: str | int | float | bool
    description: str | None = None


class AnalysisSummary(BaseModel):
    total_creatives: int = 0
    avg_ocr_confidence: float = 0.0
    avg_object_confidence: float = 0.0
    avg_topic_confidence: float = 0.0


class TopicItem(BaseModel):
    topic: str
    count: int


class AnalyticsResponse(BaseModel):
    summary: AnalysisSummary
    topics: list[TopicItem] = []
    dominant_colors: list[dict] = []
    color_class_distribution: dict[str, float] = {}
    topic_color_distribution: dict = {}
    topics_table: list[dict] = []
    total_processing_time: float = 0.0
    total_creatives_in_group: int = 0


class CreativeDetail(BaseModel):
    creative_id: str
    group_id: str
    original_filename: str
    file_path: str
    file_size: int | None = None
    file_format: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    upload_timestamp: str | None = None
    overall_status: str | None = None
    ocr_text: str | None = None
    ocr_blocks: list[dict] | None = None
    detected_objects: list[dict] | None = None
    main_topic: str | None = None
    topic_confidence: float | None = None
    dominant_colors: list[dict] | None = None
    secondary_colors: list[dict] | None = None
    palette_colors: dict | None = None


class GroupSummary(BaseModel):
    group_id: str
    total_creatives: int = 0
    created_at: str | None = None

    class Config:
        from_attributes = True
