from datetime import datetime
from datetime import timezone

import streamlit as st
from pages import page_analytics
from pages import page_details
from pages import page_settings
from pages import page_upload


st.set_page_config(
    page_title="Классификатор креативов",
    page_icon="",
    layout="wide",
)

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

    .authors-block {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 244px;
        padding: 14px 18px 16px;
        background: linear-gradient(to bottom, transparent, var(--background-color, #f0f2f6) 28%);
        box-sizing: border-box;
    }
    .authors-block .authors-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #aaa;
        margin-bottom: 5px;
    }
    .authors-block .author-name {
        font-size: 12px;
        color: #666;
        line-height: 1.6;
        font-weight: 500;
    }
    .authors-block .author-name span {
        display: inline-block;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: #c0bfbf;
        margin-right: 6px;
        vertical-align: middle;
        position: relative;
        top: -1px;
    }
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

st.sidebar.markdown(
    """
    <div class="authors-block">
        <div class="authors-label">Разработчики</div>
        <div class="author-name"><span></span>Белоглазова Анастасия</div>
        <div class="author-name"><span></span>Караваева Диана</div>
    </div>
    """,
    unsafe_allow_html=True,
)