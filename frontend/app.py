from datetime import datetime
from datetime import timezone

import streamlit as st
from pages import page_analytics
from pages import page_details
from pages import page_settings
from pages import page_upload


st.set_page_config(page_title="Классификатор креативов", layout="wide")

page = st.navigation(
    {
        "Меню": [
            st.Page(page_upload, title="Загрузка"),
            st.Page(page_analytics, title="Аналитика"),
            st.Page(page_details, title="Детали креатива"),
            st.Page(page_settings, title="Настройки"),
        ],
    },
)
page.run()

st.sidebar.caption(f"Последнее действие: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
