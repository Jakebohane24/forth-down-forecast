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
    venue_name: str | None = None
    venue_type: str | None = None
    roof_status: str | None = None
    country_code: str | None = None
    forecast_for: datetime | None = None
    weather_retrieved_at: datetime | None = None
    wind_mph: float | None = None
    wind_gust_mph: float | None = None
    temperature_f: float | None = None
    precipitation_probability: float | None = None
    precipitation_inches: float | None = None
    weather_code: int | None = None
    weather_source: str | None = None


class WeekPredictionsResponse(BaseModel):
    season: int
    week: int
    count: int
    predictions: list[PredictionResponse]


class ScheduleCardResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    game_id: str
    season: int
    week: int
    home_team: str
    away_team: str
    kickoff: datetime
    venue_name: str
    venue_type: str
    roof_status: str
    country_code: str
    prediction_eligible: bool


class WeekScheduleResponse(BaseModel):
    season: int
    week: int
    count: int
    games: list[ScheduleCardResponse]


class ModelResponse(BaseModel):
    version: str
    status: str
    training_seasons: list[int]
    use_wind: bool
    moneyline_confidence_threshold: float
    moneyline_minimum_odds: float
    artifact_created_at: str | None


class PerformanceSeason(BaseModel):
    season: int
    training_games: int
    prediction_count: int
    evaluated_games: int
    margin_mae: float | None
    win_accuracy: float | None
    moneyline_bets: int
    moneyline_settled: int
    moneyline_accuracy: float | None
    moneyline_roi: float | None
    standard_tier_bets: int = 0
    standard_tier_settled: int = 0
    standard_tier_accuracy: float | None = None
    standard_tier_roi: float | None = None
    high_tier_bets: int = 0
    high_tier_settled: int = 0
    high_tier_accuracy: float | None = None
    high_tier_roi: float | None = None


class PerformanceResponse(BaseModel):
    moneyline_threshold: float
    moneyline_minimum_odds: float
    pooled_bets: int
    pooled_roi: float
    seasons: list[PerformanceSeason]
