"""Validated public API response contracts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff: datetime | None
    predicted_home_score: int
    predicted_away_score: int
    predicted_winner: str
    model_win_confidence: float = Field(ge=0.5, le=1)
    moneyline_signal: bool
    home_moneyline: float | None
    away_moneyline: float | None
    moneyline_signal_odds: float | None
    model_version: str
    generated_at: datetime
    odds_retrieved_at: datetime | None
    actual_home_score: int | None = None
    actual_away_score: int | None = None
    prediction_correct: bool | None = None
    moneyline_signal_won: bool | None = None
    moneyline_signal_profit: float | None = None


class WeekPredictionsResponse(BaseModel):
    season: int
    week: int
    count: int
    predictions: list[PredictionResponse]


class ModelResponse(BaseModel):
    version: str
    status: str
    training_seasons: list[int]
    moneyline_confidence_threshold: float
    artifact_created_at: str | None


class PerformanceSeason(BaseModel):
    season: int
    training_games: int
    margin_mae: float
    win_accuracy: float
    moneyline_bets: int
    moneyline_roi: float


class PerformanceResponse(BaseModel):
    moneyline_threshold: float
    pooled_bets: int
    pooled_roi: float
    seasons: list[PerformanceSeason]
