"""
API v1 routers
"""
from fastapi import APIRouter
from app.api.v1 import health, auth, trading, strategies, analytics

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(strategies.router, prefix="/strategies", tags=["strategies"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])