"""Persist immutable prediction DataFrames for the public API."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from api.database import Base, create_database
from api.models import Prediction


def _timestamp(value):
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    return parsed.to_pydatetime()


def persist_predictions(
    predictions: pd.DataFrame,
    *,
    snapshot_id: str,
    model_version: str,
    database_url: str | None = None,
) -> int:
    """Insert a prediction snapshot without updating previous snapshots."""
    engine, session_factory = create_database(database_url)
    Base.metadata.create_all(engine)
    rows = predictions.reset_index().to_dict(orient="records")

    with session_factory.begin() as session:
        for row in rows:
            session.add(
                Prediction(
                    snapshot_id=snapshot_id,
                    game_id=str(row["game_id"]),
                    season=int(row["season"]),
                    week=int(row["week"]),
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    kickoff=_timestamp(row.get("kickoff")),
                    predicted_home_score=int(row["pred_home_score"]),
                    predicted_away_score=int(row["pred_away_score"]),
                    predicted_winner=row["predicted_winner"],
                    model_win_confidence=float(row["model_win_confidence"]),
                    moneyline_signal=bool(row["moneyline_signal"]),
                    home_moneyline=row.get("home_moneyline"),
                    away_moneyline=row.get("away_moneyline"),
                    moneyline_signal_odds=row.get("moneyline_signal_odds"),
                    model_version=model_version,
                    generated_at=_timestamp(row["prediction_timestamp"])
                    or datetime.now(UTC),
                    odds_retrieved_at=_timestamp(row.get("retrieved_at")),
                )
            )
    engine.dispose()
    return len(rows)
