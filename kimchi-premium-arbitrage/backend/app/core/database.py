"""
Database connection and session management
"""
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from app.config import settings

logger = logging.getLogger(__name__)

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database (create tables if not exist)
    """
    from app.models import Base
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Create TimescaleDB hypertables
        await create_hypertables(conn)
        
    logger.info("Database initialized successfully")


async def create_hypertables(conn):
    """
    Create TimescaleDB hypertables for time-series data
    """
    try:
        # Create extension if not exists
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        
        # Create hypertables for time-series data
        hypertables = [
            ("price_data", "timestamp"),
            ("orderbook_snapshots", "timestamp"),
            ("premium_data", "timestamp"),
        ]
        
        for table_name, time_column in hypertables:
            try:
                query = f"SELECT create_hypertable('{table_name}', '{time_column}', if_not_exists => TRUE);"
                await conn.execute(query)
                logger.info(f"Created hypertable for {table_name}")
            except Exception as e:
                logger.warning(f"Could not create hypertable for {table_name}: {e}")
                
    except Exception as e:
        logger.warning(f"TimescaleDB extension not available: {e}")


async def close_db():
    """
    Close database connections
    """
    await engine.dispose()
    logger.info("Database connections closed")