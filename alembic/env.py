"""
Alembic environment configuration.

Reads the database path from app settings so migrations always target
the same database as the running application — no manual URL sync needed.

Usage:
    alembic upgrade head          # apply all pending migrations
    alembic downgrade -1          # roll back one migration
    alembic revision --autogenerate -m "add_foo_column"
"""
from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# Make sure the app package is importable when running alembic from the
# project root (i.e. python -m alembic or just alembic).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.settings import get_settings  # noqa: E402 — must come after sys.path patch

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _db_url() -> str:
    path = get_settings().app_db_path
    return f"sqlite:///{path}"


def run_migrations_offline() -> None:
    """Run without a live DB connection — emits SQL to stdout."""
    context.configure(
        url=_db_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run with a live connection — applies migrations directly."""
    engine = create_engine(
        _db_url(),
        connect_args={"check_same_thread": False},
        poolclass=pool.StaticPool,
    )
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
