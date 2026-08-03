"""Generate, snapshot, and persist one week of production predictions."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from api.ingest import (
    persist_game_conditions,
    persist_predictions,
    unlocked_predictions,
)
from src.market import StaticMarketDataProvider
from src.pregame import build_pregame_features
from src.prediction import WeeklyPredictor
from src.schedule import nflverse_week
from src.training import NFLModel
from src.weather import enrich_schedule_with_weather


MODEL_VERSION = "production-2018-2025-v2-no-wind"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("week", type=int)
    parser.add_argument("--market-csv", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/production"),
    )
    parser.add_argument("--database-url")
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip display-only weather collection.",
    )
    args = parser.parse_args()

    schedule = nflverse_week(args.season, args.week)
    if schedule.empty:
        raise RuntimeError(f"No regular-season games found for {args.season} week {args.week}")
    forecasts = []
    if not args.skip_weather:
        schedule, forecasts = enrich_schedule_with_weather(schedule)

    model = NFLModel.load(args.model)
    features, unavailable = build_pregame_features(
        schedule,
        required_features=model.required_prediction_features(),
    )
    if not unavailable.empty:
        print(unavailable[["game_id", "reason"]].to_string(index=False))
    if features.empty:
        raise RuntimeError("No games have enough current-season history")

    provider = (
        StaticMarketDataProvider.from_csv(args.market_csv)
        if args.market_csv
        else None
    )
    predictions = WeeklyPredictor(model).predict(
        features,
        market_provider=provider,
    )
    predictions = predictions.join(
        schedule.set_index("game_id")[["kickoff"]],
        how="left",
    )
    generated_at = datetime.now(UTC)
    predictions = unlocked_predictions(predictions, now=generated_at)
    if predictions.empty:
        if forecasts:
            persist_game_conditions(forecasts, database_url=args.database_url)
        raise RuntimeError("All games are locked because kickoff is less than one hour away")
    timestamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
    snapshot_id = f"{args.season}-w{args.week:02d}-{timestamp}"
    path = Path("artifacts/predictions") / str(args.season) / f"{snapshot_id}.parquet"
    WeeklyPredictor.save_snapshot(predictions, path)
    count = persist_predictions(
        predictions,
        snapshot_id=snapshot_id,
        model_version=MODEL_VERSION,
        database_url=args.database_url,
    )
    if forecasts:
        persist_game_conditions(forecasts, database_url=args.database_url)
    print(f"Saved {count} predictions to {path} and the application database")


if __name__ == "__main__":
    main()
