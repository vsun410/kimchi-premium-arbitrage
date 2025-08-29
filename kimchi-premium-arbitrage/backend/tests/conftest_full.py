"""
Pytest configuration and fixtures - FULL VERSION
This is the complete conftest with all dependencies.
Renamed temporarily to avoid import issues in CI.
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.core.database import Base, get_db
from app.fixtures.data_generator import TestDataGenerator

# Override database URL for testing
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "sqlite+aiosqlite:///:memory:"
)

# Create test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def client() -> Generator:
    """Create test client"""
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a clean database session for each test"""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestSessionLocal() as session:
        yield session
    
    # Drop tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture(scope="function")
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client with overridden database"""
    async def override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    app.dependency_overrides.clear()

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

@pytest.fixture(scope="function")
async def test_data_generator() -> TestDataGenerator:
    """Create test data generator"""
    return TestDataGenerator(seed=42)

@pytest.fixture(scope="function")
async def populated_db(db_session: AsyncSession, test_data_generator: TestDataGenerator):
    """Database populated with test data"""
    await test_data_generator.populate_database(
        db_session,
        include_market_data=True,
        include_trades=True,
        include_backtests=True,
        include_paper_trading=True,
        include_strategies=True
    )
    yield db_session

@pytest.fixture(scope="function")
async def sample_strategy(db_session: AsyncSession, test_data_generator: TestDataGenerator):
    """Create a sample strategy"""
    strategy = test_data_generator.generate_strategy(
        name="Test Kimchi Strategy",
        strategy_type="kimchi_arbitrage"
    )
    db_session.add(strategy)
    await db_session.commit()
    await db_session.refresh(strategy)
    return strategy

@pytest.fixture(scope="function")
async def sample_backtest(db_session: AsyncSession, sample_strategy):
    """Create a sample backtest"""
    from app.models.backtesting import Backtest
    import json
    
    backtest = Backtest(
        name="Test Backtest",
        strategy_id=sample_strategy.id,
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        initial_capital=10000.0,
        parameters=json.dumps({
            'stop_loss': 0.02,
            'take_profit': 0.05,
            'position_size': 0.25
        }),
        status='pending'
    )
    db_session.add(backtest)
    await db_session.commit()
    await db_session.refresh(backtest)
    return backtest

@pytest.fixture(scope="function")
async def sample_paper_session(db_session: AsyncSession, test_data_generator: TestDataGenerator):
    """Create a sample paper trading session"""
    session = test_data_generator.generate_paper_session(
        name="Test Paper Session",
        active=True
    )
    db_session.add(session)
    await db_session.commit()
    await db_session.refresh(session)
    return session

# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.api = pytest.mark.api