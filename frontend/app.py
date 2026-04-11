from datetime import datetime
from datetime import timezone

import streamlit as st
from pages import page_analytics
from pages import page_details
from pages import page_settings
from pages import page_upload


st.set_page_config(
    page_title="Классификатор креативов",
    page_icon="🎨",
    layout="wide",
)

# Лёгкий полиш: чуть приятнее кнопки, метрики и заголовки.
# Без изменения цветовой схемы Streamlit и графиков.
st.markdown(
    """
    <style>
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: transform .08s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    [data-testid="stMetric"] {
        background: rgba(0,0,0,0.025);
        border-radius: 10px;
        padding: 10px 14px;
    }
    h1, h2, h3 { letter-spacing: -0.01em; }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.navigation(
    {
        "Меню": [
            st.Page(page_upload, title="Загрузка", icon="📤"),
            st.Page(page_analytics, title="Аналитика", icon="📊"),
            st.Page(page_details, title="Детали креатива", icon="🔍"),
            st.Page(page_settings, title="Настройки", icon="⚙️"),
        ],
    },
)
page.run()

st.sidebar.caption(f"Последнее действие: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
