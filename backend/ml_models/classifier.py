import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from config import NUM_COCO
from config import TOPIC_LABELS
from config import settings
from ml_models.preprocessing import clean_text_for_bert
from ml_models.preprocessing import yolo_to_vector_for_bert
from transformers import BertModel
from transformers import BertTokenizer


logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

MAX_TOKEN_LENGTH = 160
BERT_HIDDEN_SIZE = 768


class MultiModalBertClassifier(nn.Module):
    """Мультимодальный классификатор: BERT + YOLO-вектор."""

    def __init__(self, num_labels: int, yolo_dim: int = NUM_COCO):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.fc1 = nn.Linear(BERT_HIDDEN_SIZE + yolo_dim, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask, yolo_vector):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(
            mask_expanded.sum(dim=1), min=1e-9,
        )
        combined = torch.cat([pooled, yolo_vector], dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x)


def get_bert_model_and_tokenizer():
    """Загружает модель и токенизатор (singleton)."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    model_path = Path(settings.MODEL_CACHE_DIR) / settings.BERT_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Модель BERT не найдена: {model_path}")

    device = torch.device(settings.DEVICE)
    num_labels = len(TOPIC_LABELS)

    _model = MultiModalBertClassifier(num_labels=num_labels)
    _model.load_state_dict(torch.load(str(model_path), map_location=device))
    _model.to(device)
    _model.eval()

    _tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    logger.info("BERT-классификатор загружен на %s", device)

    return _model, _tokenizer


def classify_creative(ocr_text: str | None, detected_objects: list[dict] | None) -> tuple[str, float]:
    """Классифицирует креатив. Возвращает (topic, confidence)."""
    model, tokenizer = get_bert_model_and_tokenizer()
    device = torch.device(settings.DEVICE)

    # Подготовка текста
    text = clean_text_for_bert(ocr_text or "")

    # Подготовка YOLO-вектора
    classes = []
    confs = []
    if detected_objects:
        for obj in detected_objects:
            classes.append(obj.get("class", ""))
            confs.append(obj.get("confidence", 0.0))

    yolo_vec = yolo_to_vector_for_bert(classes, confs)

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

    # Инференс
    with torch.no_grad():
        logits = model(input_ids, attention_mask, yolo_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    topic = TOPIC_LABELS[pred_idx]
    logger.info("Классификация: topic=%s, confidence=%.4f", topic, confidence)

    return topic, round(confidence, 4)
