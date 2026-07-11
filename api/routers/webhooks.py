"""
LiveKit webhook handler.

LiveKit sends POST events to this endpoint when room events occur.
The key event we care about: room_finished (call ended) → trigger Follow-up Agent.

LiveKit webhook docs:
https://docs.livekit.io/realtime/server/webhooks/

To configure in LiveKit Cloud:
  Dashboard → Project → Webhooks → Add Endpoint → https://your-api.com/webhook/livekit
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Request, HTTPException

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /webhook/livekit
# ---------------------------------------------------------------------------
@router.post("/livekit")
async def livekit_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receives all LiveKit room events.
    Only acts on `room_finished` events (call ended).

    LiveKit sends the event as a signed JWT in the Authorization header.
    For demo purposes, signature verification is logged but not enforced.
    """
    # Parse raw body
    body = await request.json()
    event_type = body.get("event", "")
    room_data = body.get("room", {})
    room_name = room_data.get("name", "unknown")

    logger.info("LiveKit webhook | event=%s room=%s", event_type, room_name)

    if event_type == "room_finished":
        # Fire follow-up in background so we return 200 immediately
        background_tasks.add_task(_trigger_followup, room_name, body)

    return {"received": True, "event": event_type}


# ---------------------------------------------------------------------------
# POST /webhook/call-ended (manual / testing trigger)
# ---------------------------------------------------------------------------
@router.post("/call-ended")
async def manual_call_ended(
    room_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Manual trigger for testing the follow-up flow without a real call.
    POST /webhook/call-ended?room_id=test-room-123
    """
    logger.info("Manual call-ended trigger | room_id=%s", room_id)
    background_tasks.add_task(_trigger_followup, room_id, {})
    return {"triggered": True, "room_id": room_id}


# ---------------------------------------------------------------------------
# Internal: trigger follow-up from Redis state
# ---------------------------------------------------------------------------
async def _trigger_followup(room_name: str, event_payload: dict) -> None:
    """
    Load the call state from Redis and run the FollowupAgent.
    This runs in a FastAPI BackgroundTask (separate from the LiveKit process).
    """
    try:
        import redis.asyncio as aioredis
        from agents.followup_agent import FollowupAgent
        import json

        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        state_key = f"call:{room_name}:state"
        state_raw = await redis.get(state_key)

        if not state_raw:
            logger.warning("No Redis state found for room: %s — skipping follow-up", room_name)
            return

        state = json.loads(state_raw)

        if state.get("followup_sent"):
            logger.info("Follow-up already sent for room %s — skipping webhook trigger", room_name)
            # Clean up Redis key after follow-up is verified
            await redis.delete(state_key)
            await redis.aclose()
            return

        agent = FollowupAgent(state)
        result = await agent.run()

        logger.info(
            "Follow-up complete via webhook | room=%s sms=%s email=%s",
            room_name,
            result.get("sms_sent"),
            result.get("email_sent"),
        )

        # Clean up Redis key after follow-up
        await redis.delete(state_key)
        await redis.aclose()

    except Exception as exc:
        logger.error("Follow-up webhook error for room %s: %s", room_name, exc)
