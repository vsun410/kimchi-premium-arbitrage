"""
Simplified pytest configuration for CI testing
"""
import pytest
from fastapi.testclient import TestClient

# Create a simple test client without database dependencies
@pytest.fixture(scope="module")
def client():
    """Create test client without database"""
    # Import here to avoid circular imports
    from app.main import app
    
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