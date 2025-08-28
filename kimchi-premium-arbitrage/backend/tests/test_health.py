"""
Tests for health check endpoints
"""
import pytest
from fastapi.testclient import TestClient


def test_health_check(client: TestClient):
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "app" in data
    assert "version" in data
    assert "environment" in data


def test_ready_check(client: TestClient):
    """Test readiness check endpoint"""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert "ready" in data
    assert "checks" in data
    assert "timestamp" in data
    assert isinstance(data["checks"], dict)


def test_ping(client: TestClient):
    """Test ping endpoint"""
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ping"] == "pong"


def test_metrics(client: TestClient):
    """Test metrics endpoint"""
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    
    data = response.json()
    assert "timestamp" in data
    assert "system" in data
    assert "cpu" in data
    assert "memory" in data
    assert "disk" in data


def test_cors_headers(client: TestClient):
    """Test CORS headers are present"""
    response = client.options("/api/v1/health")
    assert response.status_code == 200
    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers


def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"
    assert data["docs"] == "/api/docs"
    assert data["health"] == "/api/v1/health"


def test_swagger_docs(client: TestClient):
    """Test Swagger documentation is accessible"""
    response = client.get("/api/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_openapi_schema(client: TestClient):
    """Test OpenAPI schema is accessible"""
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    
    data = response.json()
    assert "openapi" in data
    assert "info" in data
    assert "paths" in data