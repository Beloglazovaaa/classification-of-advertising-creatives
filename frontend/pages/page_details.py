import logging

import pandas as pd
import streamlit as st
from components.color_block import color_block_horizontal
from components.visualizer import draw_bounding_boxes
from config import MINIO_BUCKET
from config import MINIO_ENDPOINT
from config import MINIO_PUBLIC_URL
from config import TOPIC_TRANSLATIONS
from services.fetchers import fetch_creative_details
from services.fetchers import fetch_creatives_by_group
from services.fetchers import fetch_groups
from utils.helpers import is_image_available


logger = logging.getLogger(__name__)


def _display_creatives_list(creatives):
    if not creatives:
        st.info("В этой группе нет креативов.")
        return

    for c in creatives:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 1, 2])
        with col1:
            st.write(f"**{c['original_filename']}**")
        with col2:
            st.write(f"{c['image_width']}x{c['image_height']}")
        with col3:
            ts = c["upload_timestamp"].split(".")[0].replace("T", " ") if c["upload_timestamp"] else "—"
            st.write(ts)
        with col4:
            st.write("Готово" if c.get("analysis") else "В обработке")
        with col5:
            if st.button("Детали", key=f"details_btn_{c['creative_id']}"):
                st.session_state.selected_creative_id_from_table = c["creative_id"]
                st.rerun()


def _ensure_scheme(host_or_url: str) -> str:
    """Гарантирует, что значение начинается с http:// или https://."""
    if host_or_url.startswith(("http://", "https://")):
        return host_or_url.rstrip("/")
    return f"http://{host_or_url}".rstrip("/")


def _build_minio_urls(file_path: str) -> tuple[str, str]:
    """Собирает два URL: для контейнера (minio:9000) и для браузера (MINIO_PUBLIC_URL).

    В БД хранится чистый object name (`grp_xxx/<id>.png`), без схемы/бакета.
    """
    internal_base = _ensure_scheme(MINIO_ENDPOINT)
    public_base = _ensure_scheme(MINIO_PUBLIC_URL)

    # Если почему-то в БД уже сохранён полный URL, используем как есть
    if file_path.startswith(("http://", "https://")):
        internal = file_path.replace(public_base, internal_base)
        return internal, file_path

    object_name = file_path.lstrip("/")
    internal_url = f"{internal_base}/{MINIO_BUCKET}/{object_name}"
    public_url = f"{public_base}/{MINIO_BUCKET}/{object_name}"
    return internal_url, public_url


def _display_image_with_boxes(data):
    file_path = data.get("file_path") or ""
    internal_url, public_url = _build_minio_urls(file_path)

    if is_image_available(internal_url):
        try:
            image_with_boxes = draw_bounding_boxes(
                image_path_or_url=internal_url,
                ocr_blocks=data.get("ocr_blocks", []),
                detected_objects=data.get("detected_objects", []),
            )
            st.image(image_with_boxes, width=600, caption="Анализ: OCR (зелёные) и объекты (жёлтые)")
        except Exception as e:
            logger.exception("Ошибка при отрисовке")
            st.error(f"Ошибка при отрисовке: {e}")
            st.image(public_url, width=300, caption="Оригинал")
    else:
        logger.warning("Изображение недоступно: %s (public=%s)", internal_url, public_url)
        st.warning("Изображение недоступно")


def _display_basic_info(data):
    st.write(f"**Файл:** {data['original_filename']}")
    st.write(f"**Размер:** {data['file_size']} байт")
    st.write(f"**Формат:** {data['file_format']}")
    st.write(f"**Разрешение:** {data['image_width']}x{data['image_height']}")

    ts = data.get("upload_timestamp", "")
    if ts:
        ts = ts.split(".")[0].replace("T", " ")
    st.write(f"**Дата загрузки:** {ts}")

    orig_topic = data.get("main_topic", "—")
    translated = TOPIC_TRANSLATIONS.get(orig_topic, orig_topic) if orig_topic != "—" else "—"
    st.write(f"**Основная тема:** {translated}")

    confidence = data.get("topic_confidence")
    st.write(f"**Уверенность:** {round(confidence, 3) if confidence else '—'}")


def _display_ocr_info(data):
    ocr_text = data.get("ocr_text")
    if not ocr_text:
        st.info("Текст не распознан.")
        return

    st.subheader("Распознанный текст")
    st.text_area("OCR", ocr_text, height=150)

    ocr_blocks = data.get("ocr_blocks", [])
    if ocr_blocks:
        for block in ocr_blocks:
            if "bbox" in block and isinstance(block["bbox"], list):
                block["bbox"] = [round(x, 4) for x in block["bbox"]]
        st.write("Блоки текста:")
        st.dataframe(pd.DataFrame(ocr_blocks, columns=["text", "bbox", "confidence"]))
    else:
        st.info("Текстовые блоки отсутствуют.")


def _display_detection_info(data):
    detected_objects = data.get("detected_objects", [])
    if detected_objects:
        for obj in detected_objects:
            if "bbox" in obj and isinstance(obj["bbox"], list):
                obj["bbox"] = [round(x, 4) for x in obj["bbox"]]
        st.subheader("Обнаруженные объекты")
        st.dataframe(pd.DataFrame(detected_objects, columns=["class", "bbox", "confidence"]))
    else:
        st.info("Объекты не обнаружены.")


def _display_color_info(data):
    dominant_colors = data.get("dominant_colors", [])
    secondary_colors = data.get("secondary_colors", [])
    palette_colors = data.get("palette_colors", {})

    if not (dominant_colors or secondary_colors or palette_colors):
        st.info("Цвета не определены.")
        return

    st.subheader("Цвета")

    if dominant_colors:
        color_block_horizontal(dominant_colors, "Доминирующие цвета", show_percent=True, show_rgb=True)

    if secondary_colors:
        color_block_horizontal(secondary_colors, "Второстепенные цвета", show_percent=True, show_rgb=True)

    if palette_colors:
        palette_list = [
            {"hex": info["hex"], "percent": info["percent"], "class_name": cls}
            for cls, info in palette_colors.items()
        ]
        color_block_horizontal(palette_list, "По палитре", show_percent=True, show_rgb=True)


def page_details():
    st.header("Детали креатива")

    if "selected_creative_id_from_table" not in st.session_state:
        st.session_state.selected_creative_id_from_table = None

    groups = fetch_groups()
    if not groups:
        st.info("Нет доступных групп")
        return

    groups_sorted = sorted(groups, key=lambda x: x["group_id"], reverse=True)
    group_display_map = {g["group_id"]: g["display_name"] for g in groups_sorted}
    group_ids = list(group_display_map.keys())

    selected_group = st.selectbox(
        "Выберите группу",
        options=group_ids,
        format_func=lambda gid: group_display_map[gid],
        index=0,
        key="selected_group_details",
    )

    if not selected_group:
        st.session_state.selected_creative_id_from_table = None
        return

    with st.spinner("Загрузка креативов..."):
        creatives = fetch_creatives_by_group(selected_group)

    _display_creatives_list(creatives)
    st.divider()

    selected_creative_id = st.session_state.selected_creative_id_from_table

    if selected_creative_id:
        if st.button("Назад к списку"):
            st.session_state.selected_creative_id_from_table = None
            st.rerun()

        st.success(f"Выбран креатив: {selected_creative_id}")

        with st.spinner("Загрузка деталей..."):
            data = fetch_creative_details(selected_creative_id)

        if not data:
            st.error("Не удалось загрузить данные креатива.")
            if st.button("Повторить"):
                st.rerun()
            return

        if data.get("overall_status") != "SUCCESS":
            st.warning("Анализ ещё не завершён или произошла ошибка.")
            if st.button("Повторить"):
                st.rerun()
            st.image(data.get("file_path"), caption="Оригинал", width=300)
            st.write(f"**Файл:** {data.get('original_filename', 'N/A')}")
            st.write(f"**Размер:** {data.get('file_size', 'N/A')} байт")
            return

        st.divider()
        st.subheader(f"Детали креатива: {selected_creative_id}")

        _display_image_with_boxes(data)
        _display_basic_info(data)
        _display_ocr_info(data)
        _display_detection_info(data)
        _display_color_info(data)
