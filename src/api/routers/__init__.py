from fastapi import APIRouter
from src.api.routers import auth
from src.api.routers import documentation
from src.api.routers import github

router = APIRouter()

router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
router.include_router(documentation.router, prefix="/docs", tags=["Documentation"])
router.include_router(github.router, prefix="/github", tags=["GitHub Integration"])
