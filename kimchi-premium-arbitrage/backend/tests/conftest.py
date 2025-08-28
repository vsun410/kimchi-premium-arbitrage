"""
Pytest configuration and fixtures
"""
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client() -> Generator:
    """Create test client"""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def test_user():
    """Create test user data"""
    return {
        "email": "test@example.com",
        "password": "Test123!",
        "full_name": "Test User"
    }


@pytest.fixture(scope="module")
def test_token():
    """Create test token"""
    # TODO: Implement actual token generation
    return "test_token_123"