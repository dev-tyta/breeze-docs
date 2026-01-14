from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Setup logging (ensure this matches your project's logging config)
from src.utils.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Breeze-Docs API",
    description="API for generating and managing documentation with Breeze-Docs.",
    version="1.0.0"
)

# CORS (Allow all for now, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok", "service": "breeze-docs"}

# Import main router
from src.api.routers import router as api_router

# Include Routers
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
