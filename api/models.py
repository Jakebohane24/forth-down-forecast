"""Relational persistence models for versioned weekly predictions."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(80), nullable=False)
    game_id: Mapped[str] = mapped_column(String(40), nullable=False)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team: Mapped[str] = mapped_column(String(8), nullable=False)
    away_team: Mapped[str] = mapped_column(String(8), nullable=False)
    kickoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    predicted_home_score: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_away_score: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_winner: Mapped[str] = mapped_column(String(8), nullable=False)
    model_win_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    moneyline_signal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    home_moneyline: Mapped[float | None] = mapped_column(Float)
    away_moneyline: Mapped[float | None] = mapped_column(Float)
    moneyline_signal_odds: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    odds_retrieved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (
        Index(
            "uq_prediction_snapshot_game",
            "snapshot_id",
            "game_id",
            unique=True,
        ),
        Index("ix_predictions_season_week", "season", "week"),
    )


class GameResult(Base):
    """Final game outcome stored independently from prediction snapshots."""

    __tablename__ = "game_results"

    game_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team: Mapped[str] = mapped_column(String(8), nullable=False)
    away_team: Mapped[str] = mapped_column(String(8), nullable=False)
    home_score: Mapped[int] = mapped_column(Integer, nullable=False)
    away_score: Mapped[int] = mapped_column(Integer, nullable=False)
    completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    __table_args__ = (Index("ix_game_results_season_week", "season", "week"),)


class GameCondition(Base):
    """Latest pregame weather, mutable without altering the locked prediction."""

    __tablename__ = "game_conditions"

    game_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    venue_name: Mapped[str] = mapped_column(String(120), nullable=False)
    venue_type: Mapped[str] = mapped_column(String(20), nullable=False)
    roof_status: Mapped[str] = mapped_column(String(20), nullable=False)
    country_code: Mapped[str] = mapped_column(
        String(2), nullable=False, default="US"
    )
    forecast_for: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    wind_mph: Mapped[float | None] = mapped_column(Float)
    wind_gust_mph: Mapped[float | None] = mapped_column(Float)
    temperature_f: Mapped[float | None] = mapped_column(Float)
    precipitation_probability: Mapped[float | None] = mapped_column(Float)
    precipitation_inches: Mapped[float | None] = mapped_column(Float)
    weather_code: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(40), nullable=False)


class ScheduledGame(Base):
    """Upcoming schedule metadata, kept separate from model predictions."""

    __tablename__ = "scheduled_games"

    game_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    home_team: Mapped[str] = mapped_column(String(8), nullable=False)
    away_team: Mapped[str] = mapped_column(String(8), nullable=False)
    kickoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    venue_name: Mapped[str] = mapped_column(String(120), nullable=False)
    venue_type: Mapped[str] = mapped_column(String(20), nullable=False)
    roof_status: Mapped[str] = mapped_column(String(20), nullable=False)
    country_code: Mapped[str] = mapped_column(String(2), nullable=False)
    prediction_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)

    __table_args__ = (Index("ix_scheduled_games_season_week", "season", "week"),)
