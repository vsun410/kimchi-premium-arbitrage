"""
Redis connection and cache management
"""
import json
import logging
from typing import Optional, Any
from redis import asyncio as aioredis
from app.config import settings

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis connection manager"""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = await aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50,
            )
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Disconnected from Redis")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis:
            return None
        try:
            return await self.redis.get(key)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = None) -> bool:
        """Set value in cache"""
        if not self.redis:
            return False
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            if expire:
                await self.redis.setex(key, expire, value)
            else:
                await self.redis.set(key, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self.redis:
            return False
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def publish(self, channel: str, message: Any) -> bool:
        """Publish message to channel"""
        if not self.redis:
            return False
        try:
            if isinstance(message, (dict, list)):
                message = json.dumps(message)
            await self.redis.publish(channel, message)
            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False
    
    async def subscribe(self, channels: list) -> aioredis.client.PubSub:
        """Subscribe to channels"""
        if not self.redis:
            return None
        try:
            self.pubsub = self.redis.pubsub()
            await self.pubsub.subscribe(*channels)
            return self.pubsub
        except Exception as e:
            logger.error(f"Redis subscribe error: {e}")
            return None
    
    async def get_json(self, key: str) -> Optional[dict]:
        """Get JSON value from cache"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    async def set_json(self, key: str, value: dict, expire: int = None) -> bool:
        """Set JSON value in cache"""
        return await self.set(key, json.dumps(value), expire)
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        if not self.redis:
            return None
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return None
    
    async def get_list(self, key: str, start: int = 0, end: int = -1) -> list:
        """Get list from cache"""
        if not self.redis:
            return []
        try:
            return await self.redis.lrange(key, start, end)
        except Exception as e:
            logger.error(f"Redis get_list error: {e}")
            return []
    
    async def push_list(self, key: str, value: Any, max_length: int = None) -> bool:
        """Push value to list"""
        if not self.redis:
            return False
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            await self.redis.lpush(key, value)
            
            if max_length:
                await self.redis.ltrim(key, 0, max_length - 1)
            
            return True
        except Exception as e:
            logger.error(f"Redis push_list error: {e}")
            return False


# Global Redis manager instance
redis_manager = RedisManager()


async def get_redis() -> RedisManager:
    """Dependency to get Redis manager"""
    return redis_manager