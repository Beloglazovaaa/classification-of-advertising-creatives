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

    [data-testid="stSidebarNav"] a {
        font-size: 14px;
        font-weight: 400;
        letter-spacing: 0.01em;
        color: #555 !important;
    }
    [data-testid="stSidebarNav"] a:hover,
    [data-testid="stSidebarNav"] a[aria-selected="true"] {
        color: #111 !important;
        font-weight: 600;
    }

    .authors-block {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 244px;
        padding: 12px 18px 18px;
        box-sizing: border-box;
        background: linear-gradient(
            to bottom,
            transparent 0%,
            var(--background-color, #f0f2f6) 35%
        );
    }
    .authors-block .authors-rule {
        height: 1px;
        background: linear-gradient(to right, transparent, #d0d0d0, transparent);
        margin-bottom: 10px;
    }
    .authors-block .authors-label {
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #c0c0c0;
        margin-bottom: 5px;
        font-family: sans-serif;
    }
    .authors-block .authors-names {
        font-size: 12px;
        font-style: italic;
        font-weight: 300;
        color: #888;
        line-height: 1.5;
        font-family: Georgia, serif;
        letter-spacing: 0.01em;
    }
    .authors-block .authors-sep {
        color: #ccc;
        font-style: normal;
        padding: 0 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.navigation(
    {
        "Меню": [
            st.Page(page_upload,    title="Загрузка"),
            st.Page(page_analytics, title="Аналитика"),
            st.Page(page_details,   title="Детали креатива"),
            st.Page(page_settings,  title="Настройки"),
        ],
    },
)
page.run()

st.sidebar.caption(
    f"Последнее действие: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC",
)

st.sidebar.markdown(
    """
    <div class="authors-block">
        <div class="authors-rule"></div>
        <div class="authors-label">Разработчики</div>
        <div class="authors-names">
            Белоглазова А.
            <span class="authors-sep">·</span>
            Караваева Д.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)