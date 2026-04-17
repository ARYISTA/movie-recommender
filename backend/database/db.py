"""
database/db.py — SQLAlchemy async engine + session factory
FastAPI endpoints use get_db() as a dependency.
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import event
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import DATABASE_URL
from backend.database.models import Base


# ── Engine ─────────────────────────────────────────────────────────────────────
# echo=True logs every SQL statement — helpful during development, turn off in prod
engine = create_async_engine(DATABASE_URL, echo=False, future=True)

# ── Session factory ────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # keep objects usable after commit
)


async def init_db() -> None:
    """Create all tables if they don't exist. Called at app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables initialised.")


async def get_db():
    """
    FastAPI dependency — yields a session per request, always closes it.

    Usage in a route:
        async def my_route(db: AsyncSession = Depends(get_db)):
            ...
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
