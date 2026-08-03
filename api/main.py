"""FastAPI application serving predictions and transparent model results."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import distinct, select, text
from sqlalchemy.orm import Session

from api.database import Base, create_database, migrate_game_condition_columns
from api.models import GameCondition, GameResult, Prediction, ScheduledGame
from api.schemas import (
    HealthResponse,
    ModelResponse,
    PerformanceResponse,
    PerformanceSeason,
    WeekScheduleResponse,
    WeekPredictionsResponse,
)
from src.config import BettingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = PROJECT_ROOT / "reports" / "production_model.json"
ROLLING_REPORT = (
    PROJECT_ROOT / "reports" / "rolling_backtest_no_wind_2022_2025.json"
)


def _winning_profit(american_odds: float) -> float:
    return american_odds / 100 if american_odds > 0 else 100 / abs(american_odds)


def _prediction_response(
    prediction: Prediction,
    result: GameResult | None,
    condition: GameCondition | None,
):
    values = {
        column.name: getattr(prediction, column.name)
        for column in Prediction.__table__.columns
        if column.name != "id"
    }
    if condition is not None:
        values.update(
            {
                "venue_name": condition.venue_name,
                "venue_type": condition.venue_type,
                "roof_status": condition.roof_status,
                "country_code": condition.country_code,
                "forecast_for": condition.forecast_for,
                "weather_retrieved_at": condition.retrieved_at,
                "wind_mph": condition.wind_mph,
                "wind_gust_mph": condition.wind_gust_mph,
                "temperature_f": condition.temperature_f,
                "precipitation_probability": condition.precipitation_probability,
                "precipitation_inches": condition.precipitation_inches,
                "weather_code": condition.weather_code,
                "weather_source": condition.source,
            }
        )
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
        migrate_game_condition_columns(engine)
        yield
        engine.dispose()

    application = FastAPI(
        title="Fourth Down Forecast API",
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
    def health(
        session: Session = Depends(get_session),
    ) -> HealthResponse:
        session.execute(text("SELECT 1"))
        return HealthResponse(status="ok", service="fourth-down-forecast-api")

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
            latest_by_game = {}
            for row in rows:
                latest_by_game.setdefault(row.game_id, row)
            rows = sorted(
                latest_by_game.values(),
                key=lambda row: (
                    row.kickoff is None,
                    row.kickoff or row.generated_at,
                    row.game_id,
                ),
            )
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
        conditions = (
            {
                condition.game_id: condition
                for condition in session.scalars(
                    select(GameCondition).where(GameCondition.game_id.in_(game_ids))
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
                _prediction_response(
                    row,
                    results.get(row.game_id),
                    conditions.get(row.game_id),
                )
                for row in rows
            ],
        )

    @application.get(
        "/schedule/{season}/{week}",
        response_model=WeekScheduleResponse,
    )
    def weekly_schedule(
        season: int,
        week: int,
        session: Session = Depends(get_session),
    ) -> WeekScheduleResponse:
        rows = list(
            session.scalars(
                select(ScheduledGame)
                .where(
                    ScheduledGame.season == season,
                    ScheduledGame.week == week,
                    ScheduledGame.prediction_eligible.is_(True),
                )
                .order_by(ScheduledGame.kickoff, ScheduledGame.game_id)
            )
        )
        return WeekScheduleResponse(
            season=season,
            week=week,
            count=len(rows),
            games=rows,
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
            use_wind=metadata["use_wind"],
            moneyline_confidence_threshold=metadata[
                "moneyline_confidence_threshold"
            ],
            moneyline_minimum_odds=metadata["moneyline_minimum_odds"],
            artifact_created_at=metadata.get("trained_at"),
        )

    @application.get("/performance", response_model=PerformanceResponse)
    def performance(
        session: Session = Depends(get_session),
    ) -> PerformanceResponse:
        report = json.loads(ROLLING_REPORT.read_text())
        pooled = report["pooled_results"]["moneyline_signal"]
        seasons = [
            PerformanceSeason(
                season=row["test_season"],
                training_games=row["training_games"],
                prediction_count=row["test_games"],
                evaluated_games=row["test_games"],
                margin_mae=row["margin_mae"],
                win_accuracy=row["win_accuracy"],
                moneyline_bets=row["moneyline_signal"]["bets"],
                moneyline_settled=row["moneyline_signal"]["bets"],
                moneyline_accuracy=row["moneyline_signal"]["win_rate"],
                moneyline_roi=row["moneyline_signal"]["roi"],
            )
            for row in report["seasons"]
        ]
        production_season = max(
            json.loads(MODEL_MANIFEST.read_text())["training_seasons"]
        ) + 1
        prediction_rows = list(
            session.scalars(
                select(Prediction)
                .where(Prediction.season == production_season)
                .order_by(Prediction.generated_at.desc())
            )
        )
        latest_by_game = {}
        for row in prediction_rows:
            latest_by_game.setdefault(row.game_id, row)
        latest = list(latest_by_game.values())
        game_ids = list(latest_by_game)
        results = (
            {
                result.game_id: result
                for result in session.scalars(
                    select(GameResult).where(
                        GameResult.game_id.in_(game_ids),
                        GameResult.completed.is_(True),
                    )
                )
            }
            if game_ids
            else {}
        )
        evaluated = [
            (prediction, results[prediction.game_id])
            for prediction in latest
            if prediction.game_id in results
        ]
        errors = [
            abs(
                (prediction.predicted_home_score - prediction.predicted_away_score)
                - (result.home_score - result.away_score)
            )
            for prediction, result in evaluated
        ]
        correct = [
            (
                prediction.predicted_winner == result.home_team
                if result.home_score > result.away_score
                else prediction.predicted_winner == result.away_team
                if result.away_score > result.home_score
                else False
            )
            for prediction, result in evaluated
        ]
        signals = [prediction for prediction in latest if prediction.moneyline_signal]
        settled_signals = [
            (prediction, results[prediction.game_id])
            for prediction in signals
            if prediction.game_id in results
        ]
        signal_wins = [
            (
                prediction.predicted_winner == result.home_team
                if result.home_score > result.away_score
                else prediction.predicted_winner == result.away_team
                if result.away_score > result.home_score
                else False
            )
            for prediction, result in settled_signals
        ]
        returns = [
            (
                _winning_profit(prediction.moneyline_signal_odds)
                if won and prediction.moneyline_signal_odds is not None
                else -1.0
            )
            for (prediction, _), won in zip(settled_signals, signal_wins)
        ]
        prior = report["seasons"][-1]
        seasons.append(
            PerformanceSeason(
                season=production_season,
                training_games=prior["training_games"] + prior["test_games"],
                prediction_count=len(latest),
                evaluated_games=len(evaluated),
                margin_mae=sum(errors) / len(errors) if errors else None,
                win_accuracy=sum(correct) / len(correct) if correct else None,
                moneyline_bets=len(signals),
                moneyline_settled=len(settled_signals),
                moneyline_accuracy=(
                    sum(signal_wins) / len(signal_wins)
                    if signal_wins
                    else None
                ),
                moneyline_roi=(
                    sum(returns) / len(returns)
                    if returns
                    else None
                ),
            )
        )
        return PerformanceResponse(
            moneyline_threshold=BettingConfig().moneyline_confidence_threshold,
            moneyline_minimum_odds=BettingConfig().moneyline_minimum_odds,
            pooled_bets=pooled["bets"],
            pooled_roi=pooled["roi"],
            seasons=seasons,
        )

    return application


app = create_app()
