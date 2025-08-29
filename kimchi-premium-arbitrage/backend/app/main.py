"""
Main FastAPI application
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app

from app.config import settings
from app.core.logging import setup_logging
from app.core.database import init_db, close_db
from app.core.redis import redis_manager
from app.middleware.cors import setup_cors
from app.middleware.error_handler import setup_error_handlers
from app.middleware.rate_limit import setup_rate_limiting
from app.api.v1 import api_router
from app.websocket.manager import websocket_manager

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Start WebSocket manager
    await websocket_manager.startup()
    
    # Initialize database connection
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Initialize Redis connection
    try:
        await redis_manager.connect()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
    
    # TODO: Start background tasks
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Stop WebSocket manager
    await websocket_manager.shutdown()
    
    # Close database connections
    await close_db()
    
    # Close Redis connections
    await redis_manager.disconnect()
    
    # TODO: Cancel background tasks


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend API for Kimchi Premium Control Dashboard",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Setup middleware
setup_cors(app)
setup_error_handlers(app)
setup_rate_limiting(app)

# Add additional middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure as needed for production
)

# Include API routers
app.include_router(api_router, prefix="/api/v1")

# Mount WebSocket app
app.mount("/ws", websocket_manager.app)

# Mount Prometheus metrics endpoint if enabled
if settings.ENABLE_METRICS:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/api/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )