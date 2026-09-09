"""Database engine and session handling."""
from __future__ import annotations

import logging
from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings
from . import models  # noqa: F401 - registers the tables on SQLModel.metadata

log = logging.getLogger(__name__)
_settings = get_settings()

# check_same_thread=False is required because FastAPI runs sync endpoints on a
# worker thread pool. It is a SQLite-only argument.
_connect_args = (
    {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(
    _settings.database_url,
    echo=False,
    pool_pre_ping=True,
    connect_args=_connect_args,
)


def init_db() -> None:
    """Create any missing tables.

    Fine for SQLite and for a first Postgres deploy. Once the schema starts
    changing in production, put Alembic in front of this.
    """
    SQLModel.metadata.create_all(engine)
    log.info("database ready at %s", _settings.database_url.split("@")[-1])


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
