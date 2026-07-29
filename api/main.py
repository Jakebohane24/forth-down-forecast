"""FastAPI application serving predictions and transparent model results."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import distinct, select
from sqlalchemy.orm import Session

from api.database import Base, create_database
from api.models import GameResult, Prediction
from api.schemas import (
    HealthResponse,
    ModelResponse,
    PerformanceResponse,
    PerformanceSeason,
    WeekPredictionsResponse,
)
from src.config import BettingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = PROJECT_ROOT / "reports" / "production_model.json"
ROLLING_REPORT = PROJECT_ROOT / "reports" / "rolling_backtest.json"


def _winning_profit(american_odds: float) -> float:
    return american_odds / 100 if american_odds > 0 else 100 / abs(american_odds)


def _prediction_response(
    prediction: Prediction,
    result: GameResult | None,
):
    values = {
        column.name: getattr(prediction, column.name)
        for column in Prediction.__table__.columns
        if column.name != "id"
    }
    if result is None or not result.completed:
        return values

    actual_winner = (
        result.home_team
        if result.home_score > result.away_score
        else result.away_team
    )
    signal_won = (
        prediction.predicted_winner == actual_winner
        if prediction.moneyline_signal
        else None
    )
    signal_profit = None
    if signal_won is not None and prediction.moneyline_signal_odds is not None:
        signal_profit = (
            _winning_profit(prediction.moneyline_signal_odds)
            if signal_won
            else -1.0
        )
    values.update(
        {
            "actual_home_score": result.home_score,
            "actual_away_score": result.away_score,
            "prediction_correct": prediction.predicted_winner == actual_winner,
            "moneyline_signal_won": signal_won,
            "moneyline_signal_profit": signal_profit,
        }
    )
    return values


def create_app(database_url: str | None = None) -> FastAPI:
    engine, session_factory = create_database(database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        Base.metadata.create_all(engine)
        yield
        engine.dispose()

    application = FastAPI(
        title="Sunday Signal API",
        version="1.0.0",
        description="Versioned NFL score predictions and model performance.",
        lifespan=lifespan,
    )
    application.state.session_factory = session_factory
    allowed_origins = [
        value.strip()
        for value in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if value.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def get_session(request: Request):
        with request.app.state.session_factory() as session:
            yield session

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="sunday-signal-api")

    @application.get("/seasons", response_model=list[int])
    def seasons(session: Session = Depends(get_session)) -> list[int]:
        values = session.scalars(
            select(distinct(Prediction.season)).order_by(Prediction.season.desc())
        )
        return list(values)

    @application.get(
        "/predictions/{season}/{week}",
        response_model=WeekPredictionsResponse,
    )
    def weekly_predictions(
        season: int,
        week: int,
        latest_only: bool = Query(default=True),
        session: Session = Depends(get_session),
    ) -> WeekPredictionsResponse:
        statement = select(Prediction).where(
            Prediction.season == season,
            Prediction.week == week,
        )
        rows = list(
            session.scalars(
                statement.order_by(
                    Prediction.generated_at.desc(),
                    Prediction.game_id,
                )
            )
        )
        if latest_only and rows:
            latest_snapshot = rows[0].snapshot_id
            rows = [row for row in rows if row.snapshot_id == latest_snapshot]
        game_ids = [row.game_id for row in rows]
        results = (
            {
                result.game_id: result
                for result in session.scalars(
                    select(GameResult).where(GameResult.game_id.in_(game_ids))
                )
            }
            if game_ids
            else {}
        )
        return WeekPredictionsResponse(
            season=season,
            week=week,
            count=len(rows),
            predictions=[
                _prediction_response(row, results.get(row.game_id)) for row in rows
            ],
        )

    @application.get("/model", response_model=ModelResponse)
    def model_details() -> ModelResponse:
        if not MODEL_MANIFEST.exists():
            raise HTTPException(status_code=503, detail="Production model unavailable")
        metadata = json.loads(MODEL_MANIFEST.read_text())
        return ModelResponse(
            version=metadata["model_version"],
            status=metadata["status"],
            training_seasons=metadata["training_seasons"],
            moneyline_confidence_threshold=metadata[
                "moneyline_confidence_threshold"
            ],
            artifact_created_at=metadata.get("trained_at"),
        )

    @application.get("/performance", response_model=PerformanceResponse)
    def performance() -> PerformanceResponse:
        report = json.loads(ROLLING_REPORT.read_text())
        pooled = report["pooled_results"]["moneyline_62_5_confidence"]
        seasons = [
            PerformanceSeason(
                season=row["test_season"],
                training_games=row["training_games"],
                margin_mae=row["margin_mae"],
                win_accuracy=row["win_accuracy"],
                moneyline_bets=row["moneyline_62_5_confidence"]["bets"],
                moneyline_roi=row["moneyline_62_5_confidence"]["roi"],
            )
            for row in report["seasons"]
        ]
        return PerformanceResponse(
            moneyline_threshold=BettingConfig().moneyline_confidence_threshold,
            pooled_bets=pooled["bets"],
            pooled_roi=pooled["roi"],
            seasons=seasons,
        )

    return application


app = create_app()
