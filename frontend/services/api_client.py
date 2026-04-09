import logging

import requests


logger = logging.getLogger(__name__)


def get_backend_url():
    from config import BACKEND_URL
    return BACKEND_URL


def make_request(method, endpoint, **kwargs):
    url = f"{get_backend_url()}{endpoint}"
    try:
        response = requests.request(method, url, timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Ошибка запроса к %s", url)
        return None
