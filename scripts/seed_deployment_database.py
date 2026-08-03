"""Copy the portfolio-facing dataset into an empty deployment database."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from sqlalchemy import select

from api.database import Base, create_database, migrate_game_condition_columns
from api.models import GameCondition, GameResult, Prediction, ScheduledGame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"


def _values(row, *, exclude: set[str] | None = None) -> dict:
    omitted = exclude or set()
    return {
        column.name: getattr(row, column.name)
        for column in row.__table__.columns
        if column.name not in omitted
    }


def seed_database(source_url: str, target_url: str) -> dict[str, int]:
    """Idempotently copy public rows without deleting deployment data."""
    if source_url == target_url:
        raise ValueError("Source and target databases must be different")

    source_engine, source_sessions = create_database(source_url)
    target_engine, target_sessions = create_database(target_url)
    Base.metadata.create_all(target_engine)
    migrate_game_condition_columns(target_engine)

    counts = {
        "predictions": 0,
        "results": 0,
        "conditions": 0,
        "scheduled_games": 0,
    }
    with source_sessions() as source, target_sessions.begin() as target:
        predictions = list(
            source.scalars(
                select(Prediction).where(
                    (Prediction.model_version.like("%-no-wind-v2"))
                    | (Prediction.season >= 2026)
                )
            )
        )
        for row in predictions:
            exists = target.scalar(
                select(Prediction.id).where(
                    Prediction.snapshot_id == row.snapshot_id,
                    Prediction.game_id == row.game_id,
                )
            )
            if exists is None:
                target.add(Prediction(**_values(row, exclude={"id"})))
                counts["predictions"] += 1

        for model, key in (
            (GameResult, "results"),
            (GameCondition, "conditions"),
            (ScheduledGame, "scheduled_games"),
        ):
            for row in source.scalars(select(model)):
                primary_key = next(iter(model.__table__.primary_key.columns)).name
                identity = getattr(row, primary_key)
                if target.get(model, identity) is None:
                    target.add(model(**_values(row)))
                    counts[key] += 1

    source_engine.dispose()
    target_engine.dispose()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-database-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--target-database-url",
        default=os.getenv("DEPLOY_DATABASE_URL"),
    )
    args = parser.parse_args()
    if not args.target_database_url:
        raise RuntimeError(
            "Set DEPLOY_DATABASE_URL to the Neon connection string"
        )
    counts = seed_database(
        args.source_database_url,
        args.target_database_url,
    )
    print(
        "Deployment database ready: "
        + ", ".join(f"{count} new {name}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
