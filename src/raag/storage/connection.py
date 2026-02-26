"""SQLite connection management."""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker


def create_engine(db_path: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine for SQLite.

    Args:
        db_path: Path to SQLite file or async connection string.
                 If not a full URL, converts to sqlite+aiosqlite:///{path}

    Returns:
        Async SQLAlchemy engine configured for SQLite
    """
    if not db_path.startswith("sqlite"):
        db_path = f"sqlite+aiosqlite:///{db_path}"
    return create_async_engine(db_path, echo=False)


def create_session_factory(engine: AsyncEngine):
    """Create a session factory for the given engine.

    Args:
        engine: SQLAlchemy async engine

    Returns:
        Session factory that produces AsyncSession instances
    """
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
