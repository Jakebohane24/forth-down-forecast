"""Database configuration shared by the API and ingestion jobs."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"


class Base(DeclarativeBase):
    pass


def create_database(database_url: str | None = None):
    url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def migrate_game_condition_columns(engine) -> None:
    """Apply the small additive migration needed by pre-Alembic databases."""
    inspector = inspect(engine)
    if "game_conditions" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("game_conditions")}
    additions = {
        "precipitation_inches": "FLOAT",
        "weather_code": "INTEGER",
        "country_code": "VARCHAR(2) DEFAULT 'US' NOT NULL",
    }
    with engine.begin() as connection:
        for column, sql_type in additions.items():
            if column not in existing:
                connection.execute(
                    text(
                        f"ALTER TABLE game_conditions "
                        f"ADD COLUMN {column} {sql_type}"
                    )
                )
