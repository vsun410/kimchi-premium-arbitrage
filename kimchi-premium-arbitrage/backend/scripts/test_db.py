"""
Test database connection without Docker
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.core.database import engine, init_db
from app.core.redis import redis_manager
from app.models import User, Strategy


async def test_database():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        # Test database connection
        async with engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            print(f"✅ Database connection successful: {result.scalar()}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nTo fix this:")
        print("1. Make sure PostgreSQL is installed and running")
        print("2. Create a database named 'kimchi_db'")
        print("3. Create a user 'kimchi_user' with password 'kimchi_password'")
        print("4. Or run: docker-compose up -d postgres")
        return False
    
    return True


async def test_redis():
    """Test Redis connection"""
    print("\nTesting Redis connection...")
    
    try:
        await redis_manager.connect()
        await redis_manager.set("test_key", "test_value")
        value = await redis_manager.get("test_key")
        await redis_manager.delete("test_key")
        print(f"✅ Redis connection successful: test_key={value}")
        await redis_manager.disconnect()
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\nTo fix this:")
        print("1. Make sure Redis is installed and running")
        print("2. Or run: docker-compose up -d redis")
        return False
    
    return True


async def main():
    """Main test function"""
    print("=" * 50)
    print("Database Integration Test")
    print("=" * 50)
    
    db_ok = await test_database()
    redis_ok = await test_redis()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Database: {'✅ OK' if db_ok else '❌ Failed'}")
    print(f"  Redis: {'✅ OK' if redis_ok else '❌ Failed'}")
    print("=" * 50)
    
    if db_ok and redis_ok:
        print("\n🎉 All tests passed! Database integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    asyncio.run(main())