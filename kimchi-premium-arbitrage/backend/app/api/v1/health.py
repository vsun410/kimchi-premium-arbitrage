"""
Health check endpoints
"""
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, status
from app.config import settings
import psutil
import platform

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifies all dependencies are available
    In production, this would check database connections, etc.
    """
    checks = {
        "database": True,  # TODO: Implement actual DB check
        "redis": True,     # TODO: Implement actual Redis check
        "external_apis": True  # TODO: Check external API availability
    }
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    
    # Get CPU information
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Get memory information
    memory = psutil.virtual_memory()
    
    # Get disk information
    disk = psutil.disk_usage('/')
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version()
        },
        "cpu": {
            "percent": cpu_percent,
            "count": cpu_count
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
            "used": memory.used
        },
        "disk": {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent
        }
    }


@router.get("/ping", status_code=status.HTTP_200_OK)
async def ping() -> Dict[str, str]:
    """Simple ping endpoint"""
    return {"ping": "pong"}