import time
import uuid
from http import HTTPStatus

import pandas as pd
import requests
import streamlit as st
from components.styles import style_status
from components.styles import style_topic
from components.thumbnails import display_uploaded_thumbnails
from config import BACKEND_URL
from config import TOPIC_TRANSLATIONS
from services.fetchers import fetch_groups
from services.fetchers import upload_files
from utils.helpers import generate_creative_id
from utils.helpers import generate_group_id


HTTP_OK = HTTPStatus.OK


def _initialize_session_state():
    if "current_group_id" not in st.session_state:
        st.session_state.current_group_id = generate_group_id()
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())


def _handle_file_upload():
    new_uploads = st.file_uploader(
        "Выберите изображения (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
        help="Поддерживаемые форматы: JPG, PNG, WebP. Можете выбрать несколько файлов.",
    )

    if new_uploads:
        added_any = False
        existing_names = {f["name"] for f in st.session_state.selected_files}
        for file in new_uploads:
            if file.name not in existing_names:
                unique_id = str(uuid.uuid4())
                st.session_state.selected_files.append({
                    "unique_id": unique_id,
                    "name": file.name,
                    "type": file.type,
                    "size": file.size,
                    "file_obj": file,
                })
                existing_names.add(file.name)
                added_any = True

        if added_any:
            st.session_state.uploader_key = str(uuid.uuid4())
            st.rerun()


def _upload_files_to_backend():
    with st.spinner("Идёт загрузка и обработка..."):
        files_for_upload = []
        creative_ids = []
        original_filenames = []

        for file_info in st.session_state.selected_files:
            file_obj = file_info["file_obj"]
            file_obj.seek(0)
            files_for_upload.append(file_obj)
            creative_ids.append(generate_creative_id())
            original_filenames.append(file_info["name"])

        result = upload_files(files_for_upload, st.session_state.current_group_id, creative_ids, original_filenames)
        if result:
            st.success(
                f"Успешно загружено {result['uploaded']} файлов в группу {st.session_state.current_group_id}",
            )
            st.session_state.uploaded_creatives = creative_ids
            st.session_state.selected_files = []
            st.session_state.uploader_key = str(uuid.uuid4())
            st.session_state.pop("current_group_id", None)
            fetch_groups.clear()
            st.rerun()
        else:
            st.error("Ошибка загрузки")


def _get_creative_status(cid):
    try:
        resp = requests.get(f"{BACKEND_URL}/status/{cid}", timeout=10)
        if resp.status_code == HTTP_OK:
            return resp.json(), None
    except requests.exceptions.RequestException as e:
        return None, f"Сеть: {type(e).__name__}"
    except (TypeError, RuntimeError) as e:
        return None, str(e)
    else:
        return None, f"Статус {resp.status_code}"


def _process_status_data(data):
    original_topic = data["main_topic"]
    translated_topic = TOPIC_TRANSLATIONS.get(original_topic, original_topic) if original_topic else "—"

    stage_statuses = [
        data["ocr_status"],
        data["detection_status"],
        data["classification_status"],
        data["color_status"],
    ]

    is_finished = all(
        isinstance(s, str) and s.endswith("sec") and not s.endswith("sec ")
        for s in stage_statuses if s != "X"
    )

    status_entry = {
        "ID": data["creative_id"][:8] + "...",
        "Файл": data["original_filename"],
        "Размер": data["file_size"],
        "Разрешение": data["image_size"],
        "Время загрузки": data["upload_timestamp"].split(".")[0].replace("T", " "),
        "OCR": data["ocr_status"],
        "Детекция": data["detection_status"],
        "Классиф.": data["classification_status"],
        "Цвет": data["color_status"],
        "Топик": translated_topic or "PENDING",
        "Confidence": f"{data['topic_confidence']:.2f}" if data["topic_confidence"] else "—",
        "Статус": data.get("overall_status", "—"),
    }

    return status_entry, is_finished


def _display_status_table(statuses):
    df = pd.DataFrame(statuses)
    if df.empty:
        return None

    # Гарантируем наличие всех колонок (записи с ошибками дополняем "—"),
    # иначе df.style.map упадёт с KeyError при первой же ошибке статуса.
    required_cols = [
        "ID", "Файл", "Размер", "Разрешение", "Время загрузки",
        "OCR", "Детекция", "Классиф.", "Цвет", "Топик", "Confidence", "Статус",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "—"
    df = df[required_cols + [c for c in df.columns if c not in required_cols]]

    style_cols = [c for c in ["OCR", "Детекция", "Классиф.", "Цвет", "Статус"] if c in df.columns]
    topic_cols = [c for c in ["Топик"] if c in df.columns]

    styler = df.style
    if style_cols:
        styler = styler.map(style_status, subset=style_cols)
    if topic_cols:
        styler = styler.map(style_topic, subset=topic_cols)
    return styler


def _display_processing_status():
    st.subheader("Статус обработки")
    st.markdown(f"**Группа:** `{st.session_state.current_group_id}`")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.color_picker("Готово", "#69cd81", disabled=True)
    with col2:
        st.color_picker("В процессе", "#f1d477", disabled=True)
    with col3:
        st.color_picker("Ожидание", "#c4c4c4", disabled=True)
    with col4:
        st.color_picker("Ошибка", "#f38080", disabled=True)

    status_table = st.empty()

    while True:
        statuses = []
        finished_count = 0
        total_count = len(st.session_state.uploaded_creatives)

        for cid in st.session_state.uploaded_creatives:
            data, error = _get_creative_status(cid)
            if data:
                status_entry, is_finished = _process_status_data(data)
                statuses.append(status_entry)
                if is_finished:
                    finished_count += 1
            else:
                statuses.append({"ID": cid[:8] + "...", "Ошибка": error})

        styled_df = _display_status_table(statuses)
        if styled_df is not None:
            status_table.dataframe(styled_df, use_container_width=True)

        if finished_count == total_count and total_count > 0:
            st.success("Все креативы обработаны!")
            st.session_state.uploaded_creatives = []
            return

        time.sleep(1)


def page_upload():
    st.header("Загрузка креативов")

    _initialize_session_state()
    st.text(f"Текущая группа: {st.session_state.current_group_id}")

    _handle_file_upload()

    st.subheader("Выбранные файлы")
    display_uploaded_thumbnails(st.session_state.selected_files)

    if st.session_state.selected_files and st.button("Загрузить", key="upload_btn"):
        _upload_files_to_backend()

    if "uploaded_creatives" in st.session_state and st.session_state.uploaded_creatives:
        _display_processing_status()
