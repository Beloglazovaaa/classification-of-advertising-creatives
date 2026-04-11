#!/bin/sh
set -e

until mc alias set minio http://minio:9000 "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}"; do
    echo "⏳ Waiting for MinIO..."
    sleep 1
done

echo "📦 Creating buckets..."
mc mb minio/creatives --ignore-existing
mc mb minio/models --ignore-existing

# Публичный доступ только для bucket с креативами
mc anonymous set public minio/creatives

# ───── YOLO ─────
if [ -f "/minio_init/models/yolov8m.pt" ]; then
    echo "⬆️  Uploading yolov8m.pt..."
    mc cp /minio_init/models/yolov8m.pt minio/models/yolov8m.pt
else
    echo "⚠️  yolov8m.pt not found — skip"
fi

# ───── BERT classifier ─────
if [ -f "/minio_init/models/best_multimodal_bert.pt" ]; then
    echo "⬆️  Uploading best_multimodal_bert.pt..."
    mc cp /minio_init/models/best_multimodal_bert.pt minio/models/best_multimodal_bert.pt
else
    echo "⚠️  best_multimodal_bert.pt not found — skip"
fi

# ───── EasyOCR weights ─────
# Backend ждёт ключи вида: easy_ocr/craft_mlt_25k.pth (без вложенного model/)
if [ -d "/minio_init/models/easy_ocr" ]; then
    echo "⬆️  Uploading easy_ocr weights..."
    for f in craft_mlt_25k.pth cyrillic_g2.pth english_g2.pth; do
        if [ -f "/minio_init/models/easy_ocr/$f" ]; then
            mc cp "/minio_init/models/easy_ocr/$f" "minio/models/easy_ocr/$f"
        elif [ -f "/minio_init/models/easy_ocr/model/$f" ]; then
            mc cp "/minio_init/models/easy_ocr/model/$f" "minio/models/easy_ocr/$f"
        else
            echo "⚠️  easy_ocr/$f not found — skip"
        fi
    done
fi

echo "✅ MinIO initialization completed successfully"
echo "📊 Uploaded objects:"
mc ls --recursive minio/models/ || true
