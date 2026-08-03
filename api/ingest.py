"""Persist immutable prediction DataFrames for the public API."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from api.database import Base, create_database, migrate_game_condition_columns
from api.models import GameCondition, GameResult, Prediction, ScheduledGame
from src.weather import WeatherForecast


def _timestamp(value):
    if value is None or pd.isna(value):
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize(UTC)
    return parsed.to_pydatetime()


def unlocked_predictions(
    predictions: pd.DataFrame,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return only games whose one-hour prediction lock has not passed."""
    current_time = now or datetime.now(UTC)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UTC)
    current_time = current_time.astimezone(UTC)

    def is_unlocked(value) -> bool:
        kickoff = _timestamp(value)
        return (
            kickoff is None
            or current_time < kickoff.astimezone(UTC) - timedelta(hours=1)
        )

    if "kickoff" not in predictions.columns:
        return predictions.copy()
    return predictions.loc[predictions["kickoff"].map(is_unlocked)].copy()


def persist_predictions(
    predictions: pd.DataFrame,
    *,
    snapshot_id: str,
    model_version: str,
    database_url: str | None = None,
    enforce_kickoff_lock: bool = True,
    now: datetime | None = None,
) -> int:
    """Insert predictions that are still at least one hour from kickoff.

    Previous snapshots remain immutable. This per-game cutoff allows later
    games to refresh without changing picks for games whose lock has passed.
    """
    engine, session_factory = create_database(database_url)
    Base.metadata.create_all(engine)
    migrate_game_condition_columns(engine)
    eligible = (
        unlocked_predictions(predictions, now=now)
        if enforce_kickoff_lock
        else predictions
    )
    eligible_rows = eligible.reset_index().to_dict(orient="records")

    with session_factory.begin() as session:
        for row in eligible_rows:
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
    return len(eligible_rows)


def persist_game_results(
    games: pd.DataFrame,
    *,
    database_url: str | None = None,
) -> int:
    """Insert or refresh final results without altering any prediction."""
    engine, session_factory = create_database(database_url)
    Base.metadata.create_all(engine)
    rows = games.reset_index().to_dict(orient="records")
    updated_at = datetime.now(UTC)

    with session_factory.begin() as session:
        for row in rows:
            game_id = str(row["game_id"])
            result = session.get(GameResult, game_id)
            values = {
                "season": int(row["season"]),
                "week": int(row["week"]),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_score": int(row["home_score"]),
                "away_score": int(row["away_score"]),
                "completed": True,
                "updated_at": updated_at,
            }
            if result is None:
                session.add(GameResult(game_id=game_id, **values))
            else:
                for key, value in values.items():
                    setattr(result, key, value)
    engine.dispose()
    return len(rows)


def persist_game_conditions(
    forecasts: list[WeatherForecast],
    *,
    database_url: str | None = None,
) -> int:
    """Upsert the latest weather while preserving prediction snapshots."""
    engine, session_factory = create_database(database_url)
    Base.metadata.create_all(engine)
    migrate_game_condition_columns(engine)
    with session_factory.begin() as session:
        for forecast in forecasts:
            condition = session.get(GameCondition, forecast.game_id)
            values = {
                "venue_name": forecast.venue_name,
                "venue_type": forecast.venue_type,
                "roof_status": forecast.roof_status,
                "country_code": forecast.country_code,
                "forecast_for": forecast.forecast_for,
                "retrieved_at": forecast.retrieved_at,
                "wind_mph": forecast.wind_mph,
                "wind_gust_mph": forecast.wind_gust_mph,
                "temperature_f": forecast.temperature_f,
                "precipitation_probability": forecast.precipitation_probability,
                "precipitation_inches": forecast.precipitation_inches,
                "weather_code": forecast.weather_code,
                "source": forecast.source,
            }
            if condition is None:
                session.add(GameCondition(game_id=forecast.game_id, **values))
            else:
                for key, value in values.items():
                    setattr(condition, key, value)
    engine.dispose()
    return len(forecasts)


def persist_scheduled_games(
    games: pd.DataFrame,
    *,
    database_url: str | None = None,
) -> int:
    """Upsert schedule-only cards without fabricating prediction rows."""
    engine, session_factory = create_database(database_url)
    Base.metadata.create_all(engine)
    rows = games.to_dict(orient="records")
    with session_factory.begin() as session:
        for row in rows:
            game_id = str(row["game_id"])
            scheduled = session.get(ScheduledGame, game_id)
            values = {
                "season": int(row["season"]),
                "week": int(row["week"]),
                "home_team": str(row["home_team"]),
                "away_team": str(row["away_team"]),
                "kickoff": _timestamp(row["kickoff"]),
                "venue_name": str(row["venue_name"]),
                "venue_type": str(row["venue_type"]),
                "roof_status": str(row["roof_status"]),
                "country_code": str(row["country_code"]),
                "prediction_eligible": bool(row["prediction_eligible"]),
            }
            if scheduled is None:
                session.add(ScheduledGame(game_id=game_id, **values))
            else:
                for key, value in values.items():
                    setattr(scheduled, key, value)
    engine.dispose()
    return len(rows)
