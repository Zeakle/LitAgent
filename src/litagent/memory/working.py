"""Working Memory — Redis 后端。

会话级热存储。key 格式: litagent:working:{session_id}。
TTL 30 分钟，每次 get 自动续期。
"""


from redis.asyncio import Redis

from litagent.config import MemoryConfig
from litagent.logging import get_logger


logger = get_logger("memory.working")

_KEY_PREFIX = 'litagent:working:'


class WorkingMemory:
    """Redis 存储的working memory

    存储 AgentState dict, TTL自动管理
    orjson 序列化
    """

    def __init__(self, redis: Redis, config: MemoryConfig):
        self._redis = redis
        self._ttl = config.working_ttl_seconds

    @staticmethod
    async def connect(config: MemoryConfig) -> "WorkingMemory":
        redis = Redis.from_url(config.redis_url, decode_responses=False)
        await redis.ping()
        logger.info(f"Connected to Redis")  # URL may contain password, don't log it
        return WorkingMemory(redis, config)

    async def get(self, session_id: str) -> dict | None:
        """读取 session state. 不存在或过期返回None"""
        import orjson
        key = _KEY_PREFIX + session_id
        data = await self._redis.get(key)
        if data is None:
            return None

        # 续期 TTL
        await self._redis.expire(key, self._ttl)
        return orjson.loads(data)

    async def set(self, session_id: str, state: dict) -> None:
        """写入 session state, orjson序列化"""
        import orjson
        key = _KEY_PREFIX + session_id
        if 'messages' in state:
            state['messages'] = [
                m.model_dump() if hasattr(m, 'model_dump') else m for m in state['messages']
            ]
        await self._redis.set(key, orjson.dumps(state), ex=self._ttl)
    
    async def delete(self, session_id: str) -> None:
        key = _KEY_PREFIX + session_id
        await self._redis.delete(key)

    async def exists(self, session_id: str) -> bool:
        key = _KEY_PREFIX + session_id
        return await self._redis.exists(key) > 0