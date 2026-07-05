"""
Redis session store — persists TriageState during a live call.
Used by the webhook handler to retrieve state after the LiveKit process ends.
"""
from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis


async def save_state(room_id: str, state: dict, ttl_seconds: int = 7200) -> None:
    """Persist TriageState to Redis with a TTL."""
    try:
        r = get_redis()
        key = f"call:{room_id}:state"
        await r.setex(key, ttl_seconds, json.dumps(state, default=str))
    except Exception as exc:
        logger.warning("Redis save failed for room %s: %s", room_id, exc)


async def load_state(room_id: str) -> dict | None:
    """Load TriageState from Redis. Returns None if not found."""
    try:
        r = get_redis()
        key = f"call:{room_id}:state"
        raw = await r.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.warning("Redis load failed for room %s: %s", room_id, exc)
        return None


async def delete_state(room_id: str) -> None:
    """Clean up Redis key after follow-up is sent."""
    try:
        r = get_redis()
        await r.delete(f"call:{room_id}:state")
    except Exception as exc:
        logger.warning("Redis delete failed for room %s: %s", room_id, exc)
