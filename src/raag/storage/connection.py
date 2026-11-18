"""SQLite connection management."""

import re

from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession, async_sessionmaker

_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*")
# Two or more scheme characters, so a Windows drive letter is not a scheme
_URL_LIKE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]+:")


def create_engine(db_path: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine for SQLite.

    Args:
        db_path: Path to SQLite file or async connection string.
                 If not a full URL, converts to sqlite+aiosqlite:///{path}

    Returns:
        Async SQLAlchemy engine configured for SQLite

    Raises:
        ValueError: If db_path is a malformed URL, a URL for a database
            other than SQLite, or starts like a URL scheme ("name:") without
            "//". The message never includes credentials.
    """
    if "://" in db_path:
        scheme = db_path.split("://", 1)[0]
        if not _SCHEME.fullmatch(scheme):
            raise ValueError(
                "Invalid database URL. "
                "Only SQLite is supported: pass a file path or a sqlite+aiosqlite:// URL."
            )
        if not scheme.startswith("sqlite"):
            raise ValueError(
                f"Unsupported database URL scheme '{scheme}'. "
                "Only SQLite is supported: pass a file path or a sqlite+aiosqlite:// URL."
            )
    elif _URL_LIKE.match(db_path):
        raise ValueError(
            "Invalid database path: it looks like a URL without '//'. "
            "Only SQLite is supported: pass a file path (start it with ./ if its "
            "name contains a colon) or a sqlite+aiosqlite:// URL."
        )
    else:
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
