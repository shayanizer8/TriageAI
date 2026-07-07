"""
Async SQLAlchemy engine + session factory.
"""
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase
from config.settings import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.is_development,
    pool_pre_ping=True,   # validates connections before use
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def get_db() -> AsyncSession:
    """
    FastAPI dependency — yields an async DB session per request.
    Commits on success, rolls back on exception.
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


async def init_db() -> None:
    """
    Create all tables. Called at application startup.
    In production prefer Alembic migrations.
    """
    async with engine.begin() as conn:
        from db import models  # noqa: F401 — ensures models are registered
        await conn.run_sync(Base.metadata.create_all)
