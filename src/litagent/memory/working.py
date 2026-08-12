"""Store short-lived session state in Redis with sliding expiration."""

from redis.asyncio import Redis

from litagent.config import MemoryConfig
from litagent.logging import get_logger

logger = get_logger("memory.working")

_KEY_PREFIX = "litagent:working:"


class WorkingMemory:
    """Persist active session state in Redis."""

    def __init__(self, redis: Redis, config: MemoryConfig):
        """Initialize the working memory."""
        self._redis = redis
        self._ttl = config.working_ttl_seconds

    @staticmethod
    async def connect(config: MemoryConfig) -> "WorkingMemory":
        """Connect to Redis and close the client if validation cannot finish."""
        redis = Redis.from_url(config.redis_url, decode_responses=False)
        try:
            await redis.ping()
        except BaseException:
            # Ownership has not transferred to Infra until this method returns.
            try:
                await redis.aclose()
            except BaseException as close_exc:
                logger.debug("Redis rollback close error: %s", close_exc)
            raise
        # Avoid logging the Redis URL because it may contain credentials.
        logger.info("Connected to Redis")
        return WorkingMemory(redis, config)

    async def get(self, session_id: str) -> dict | None:
        """Load session state and refresh its expiration."""
        import orjson

        key = _KEY_PREFIX + session_id
        data = await self._redis.get(key)
        if data is None:
            return None

        # Refresh the sliding session TTL after every successful read.
        await self._redis.expire(key, self._ttl)
        return orjson.loads(data)

    async def set(self, session_id: str, state: dict) -> None:
        """Serialize and store session state with the configured expiration."""
        import orjson

        key = _KEY_PREFIX + session_id
        # Convert message objects to JSON-serializable mappings.
        if "messages" in state:
            state["messages"] = [
                m.model_dump() if hasattr(m, "model_dump") else m
                for m in state["messages"]
            ]
        await self._redis.set(key, orjson.dumps(state), ex=self._ttl)

    async def delete(self, session_id: str) -> None:
        """Delete session state by identifier."""
        key = _KEY_PREFIX + session_id
        await self._redis.delete(key)

    async def exists(self, session_id: str) -> bool:
        """Return whether session state exists."""
        key = _KEY_PREFIX + session_id
        return await self._redis.exists(key) > 0

    async def close(self) -> None:
        """Close the Redis client."""
        await self._redis.aclose()
