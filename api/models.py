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
