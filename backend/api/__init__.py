from fastapi import APIRouter

from .analytics import router as analytics_router
from .creatives import router as creatives_router
from .groups import router as groups_router
from .settings import router as settings_router
from .status import router as status_router
from .upload import router as upload_router


router = APIRouter()

router.include_router(groups_router)
router.include_router(upload_router)
router.include_router(creatives_router)
router.include_router(status_router)
router.include_router(analytics_router)
router.include_router(settings_router)
