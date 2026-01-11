"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.config import get_settings
from src.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    settings = get_settings()
    setup_logging(level=settings.log_level, log_file="backend.log")

    # Validate API keys
    key_status = settings.validate_api_keys()
    missing_keys = [k for k, v in key_status.items() if not v]
    if missing_keys:
        print(f"WARNING: Missing API keys for: {', '.join(missing_keys)}")
        print("Some agents may not work correctly.")

    yield

    # Shutdown
    pass


app = FastAPI(
    title="Synapse Council",
    description="Multi-AI Architecture Review Board for System Design",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Synapse Council",
        "description": "Multi-AI Architecture Review Board",
        "version": "0.1.0",
        "endpoints": {
            "start_design": "POST /api/design",
            "get_status": "GET /api/design/{session_id}",
            "respond": "POST /api/design/{session_id}/respond",
            "get_result": "GET /api/design/{session_id}/result",
            "websocket": "WS /api/ws/{session_id}",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "api_keys": settings.validate_api_keys(),
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
