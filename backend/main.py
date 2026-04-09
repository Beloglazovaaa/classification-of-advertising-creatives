import logging

from api import router
from core import lifespan
from fastapi import FastAPI


class StatusFilter(logging.Filter):
    """Фильтрует повторяющиеся GET /status/ запросы из логов."""

    def filter(self, record):
        if hasattr(record, "args") and record.args and len(record.args) >= 5:
            method = record.args[1]
            path = record.args[2]
            status_code = record.args[4]
            if method == "GET" and path.startswith("/status/") and status_code == 200:
                return False
        return True


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

uvicorn_access_logger = logging.getLogger("uvicorn.access")
if uvicorn_access_logger:
    uvicorn_access_logger.addFilter(StatusFilter())

app = FastAPI(title="Creative Classification API", lifespan=lifespan)
app.include_router(router)

logger.info("Маршруты подключены:")
for route in app.routes:
    if hasattr(route, "path"):
        methods = getattr(route, "methods", set())
        logger.info("  %s %s → %s", methods, route.path, route.name)
