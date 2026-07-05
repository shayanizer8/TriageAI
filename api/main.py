"""
FastAPI application — HTTP backend for TriageAI.

Routers:
  /health            — liveness probe
  /schedule          — Mock HIS API (doctors, slots, booking)
  /triage            — RAG query endpoint (for testing Pinecone)
  /webhook           — LiveKit post-call webhook
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from db.database import init_db

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TriageAI API | env=%s", settings.environment)
    # Create DB tables on first run (use Alembic in production)
    await init_db()
    yield
    logger.info("Shutting down TriageAI API")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TriageAI Backend",
    description="AI-powered telephone triage system — HTTP API",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
from api.routers import schedule, triage, webhooks  # noqa: E402

app.include_router(schedule.router, prefix="/schedule", tags=["Scheduling (Mock HIS)"])
app.include_router(triage.router, prefix="/triage", tags=["Triage RAG"])
app.include_router(webhooks.router, prefix="/webhook", tags=["LiveKit Webhooks"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "TriageAI API", "environment": settings.environment}
