import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from config import NUM_COCO
from config import TOPIC_LABELS
from config import settings
from config import map_coco_to_topic
from ml_models.clip_extractor import CLIP_FEATURE_DIM
from ml_models.clip_extractor import extract_clip_features
from ml_models.clip_extractor import zeros_vector as clip_zeros
from ml_models.preprocessing import clean_text_for_bert
from ml_models.preprocessing import yolo_to_vector_for_bert
from transformers import BertModel
from transformers import BertTokenizer


logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_bert_unavailable = False  # Если BERT не удалось загрузить, используем fallback

MAX_TOKEN_LENGTH = 160
BERT_HIDDEN_SIZE = 768


class MultiModalBertClassifier(nn.Module):
    """Мультимодальный классификатор: BERT(текст) + CLIP(изображение) + YOLO(детекция).

    Архитектура:
        text    →  BERT → mean-pool → 768
        image   →  CLIP (pre-computed) → 512  → proj(256)
        objects →  YOLO one-hot (80)           → proj(64)
        ─────── concat 768 + 256 + 64 = 1088 ───────
        → fc(256) → ReLU → Dropout(0.3) → fc(num_labels)
    """

    def __init__(
        self,
        num_labels: int,
        yolo_dim: int = NUM_COCO,
        clip_dim: int = CLIP_FEATURE_DIM,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        # Проекции каждой модальности (чтобы сбалансировать размерности)
        self.clip_proj = nn.Linear(clip_dim, 256)
        self.yolo_proj = nn.Linear(yolo_dim, 64)

        fusion_dim = BERT_HIDDEN_SIZE + 256 + 64
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, yolo_vector, clip_vector):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling по токенам
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_pooled = torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9,
        )

        clip_proj = torch.relu(self.clip_proj(clip_vector))
        yolo_proj = torch.relu(self.yolo_proj(yolo_vector))

        combined = torch.cat([text_pooled, clip_proj, yolo_proj], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x)


def get_bert_model_and_tokenizer():
    """Загружает модель и токенизатор (singleton). Возвращает (None, None) если недоступна."""
    global _model, _tokenizer, _bert_unavailable

    if _bert_unavailable:
        return None, None

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    try:
        model_path = Path(settings.MODEL_CACHE_DIR) / settings.BERT_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(f"Модель BERT не найдена: {model_path}")

        device = torch.device(settings.DEVICE)
        num_labels = len(TOPIC_LABELS)

        _model = MultiModalBertClassifier(num_labels=num_labels)
        state_dict = torch.load(str(model_path), map_location=device)
        # Если чекпоинт обучен на другое число классов (например, старая 5-классовая модель),
        # head не совпадёт по форме — переходим в fallback вместо краха.
        try:
            _model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(
                "Чекпоинт BERT не совпадает с текущей архитектурой (%d классов): %s. "
                "Переключаюсь на fallback. Переобучи через scripts/train.sh.",
                num_labels, e,
            )
            raise
        _model.to(device)
        _model.eval()

        _tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        logger.info("BERT-классификатор загружен на %s", device)
    except (FileNotFoundError, OSError, RuntimeError) as e:
        logger.warning("BERT недоступен (%s). Используется fallback по YOLO-классам.", e)
        _bert_unavailable = True
        _model = None
        _tokenizer = None
        return None, None

    return _model, _tokenizer


def _fallback_classify(classes: list[str], confs: list[float]) -> tuple[str, float]:
    """Fallback-классификация по COCO-классам YOLO, если BERT недоступен."""
    # Суммируем уверенность по каждому топику
    topic_scores: dict[str, float] = {}
    for cls, conf in zip(classes, confs, strict=False):
        topic = map_coco_to_topic(cls)
        if topic:
            topic_scores[topic] = topic_scores.get(topic, 0.0) + float(conf)

    if not topic_scores:
        # Нет подходящих детекций — возвращаем первый топик с низкой уверенностью
        return TOPIC_LABELS[0], 0.0

    best_topic = max(topic_scores, key=topic_scores.get)
    # Нормируем псевдо-confidence в [0, 1]
    total = sum(topic_scores.values()) or 1.0
    confidence = min(1.0, topic_scores[best_topic] / total)
    return best_topic, confidence


def classify_creative(
    ocr_text: str | None,
    detected_objects: list[dict] | None,
    image_path: str | None = None,
) -> tuple[str, float]:
    """Классифицирует креатив. Возвращает (topic, confidence).

    image_path — путь к изображению для CLIP-фичей. Если None или CLIP недоступен,
    используется нулевой вектор (модель продолжит работать, но без визуальной модальности).
    """
    model, tokenizer = get_bert_model_and_tokenizer()

    # Подготовка YOLO-классов (нужны и для BERT, и для fallback)
    classes: list[str] = []
    confs: list[float] = []
    if detected_objects:
        for obj in detected_objects:
            classes.append(obj.get("class", ""))
            confs.append(obj.get("confidence", 0.0))

    # Fallback: BERT недоступен — классифицируем по COCO-классам YOLO
    if model is None or tokenizer is None:
        topic, confidence = _fallback_classify(classes, confs)
        logger.info("Fallback-классификация: topic=%s, confidence=%.4f", topic, confidence)
        return topic, round(confidence, 4)

    device = torch.device(settings.DEVICE)

    # Подготовка текста
    text = clean_text_for_bert(ocr_text or "")

    yolo_vec = yolo_to_vector_for_bert(classes, confs)

    # CLIP-фичи изображения
    clip_vec = None
    if image_path:
        clip_vec = extract_clip_features(image_path)
    if clip_vec is None:
        clip_vec = clip_zeros()

    # Токенизация
    encoded = tokenizer(
        text,
        max_length=MAX_TOKEN_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    yolo_tensor = torch.tensor([yolo_vec], dtype=torch.float32).to(device)
    clip_tensor = torch.tensor([clip_vec], dtype=torch.float32).to(device)

    # Инференс
    with torch.no_grad():
        logits = model(input_ids, attention_mask, yolo_tensor, clip_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    topic = TOPIC_LABELS[pred_idx]
    logger.info("Классификация: topic=%s, confidence=%.4f", topic, confidence)

    return topic, round(confidence, 4)
