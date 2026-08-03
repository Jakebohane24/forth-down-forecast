"""Cache rolling models and publish honest historical prediction snapshots."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy import select

from api.database import create_database
from api.ingest import persist_game_results, persist_predictions
from api.models import Prediction
from src.config import BettingConfig, ModelConfig
from src.evaluation import evaluation_frame
from src.training import NFLModel


def snapshot_exists(snapshot_id: str, database_url: str | None) -> bool:
    engine, session_factory = create_database(database_url)
    with session_factory() as session:
        exists = session.scalar(
            select(Prediction.id)
            .where(Prediction.snapshot_id == snapshot_id)
            .limit(1)
        )
    engine.dispose()
    return exists is not None


def american_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame["predicted_winner"] = frame["home_team"].where(
        frame["pred_home_win"],
        frame["away_team"],
    )
    frame["model_win_confidence"] = frame["home_win_prob"].where(
        frame["pred_home_win"],
        1 - frame["home_win_prob"],
    )
    config = BettingConfig()
    selected_moneyline = frame["home_moneyline"].where(
        frame["pred_home_win"],
        frame["away_moneyline"],
    )
    frame["moneyline_signal"] = (
        (frame["model_win_confidence"] >= config.moneyline_confidence_threshold)
        & (selected_moneyline >= config.moneyline_minimum_odds)
    )
    frame["moneyline_signal_odds"] = selected_moneyline.where(
        frame["moneyline_signal"]
    )
    return frame


def schedule_metadata(seasons: list[int]) -> pd.DataFrame:
    schedule = nfl.import_schedules(seasons).copy()
    kickoff_eastern = pd.to_datetime(
        schedule["gameday"].astype(str)
        + " "
        + schedule["gametime"].fillna("00:00"),
        errors="coerce",
    )
    schedule["kickoff"] = kickoff_eastern.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")
    return schedule[
        [
            "game_id",
            "kickoff",
            "home_moneyline",
            "away_moneyline",
        ]
    ].set_index("game_id")


def model_for_season(test_season: int, artifact_root: Path) -> NFLModel:
    artifact = artifact_root / str(test_season)
    if (artifact / "models.joblib").exists():
        print(f"{test_season}: loading cached model")
        return NFLModel.load(artifact)

    training_seasons = tuple(range(2018, test_season))
    config = ModelConfig(
        use_wind=False,
        training_seasons=training_seasons,
        validation_season=test_season - 1,
        test_season=test_season,
    )
    print(
        f"{test_season}: training on {training_seasons[0]}–"
        f"{training_seasons[-1]}"
    )
    model = NFLModel(config).train()
    model.save(
        artifact,
        model_version=f"rolling-through-{test_season - 1}-no-wind-v3-tie-adjusted",
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-season", type=int, default=2021)
    parser.add_argument("--last-season", type=int, default=2025)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/models/no_wind_rolling"),
    )
    parser.add_argument("--database-url")
    args = parser.parse_args()

    seasons = list(range(args.first_season, args.last_season + 1))
    markets = schedule_metadata(seasons)
    generated_at = datetime.now(UTC)

    for season in seasons:
        model = model_for_season(season, args.artifact_root)
        frame = evaluation_frame(model, "test").join(markets, how="left")
        frame = american_columns(frame)
        frame["prediction_timestamp"] = generated_at
        persist_game_results(
            frame[
                [
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                ]
            ],
            database_url=args.database_url,
        )

        for week, week_frame in frame.groupby("week"):
            snapshot_id = (
                f"historical-{season}-w{int(week):02d}-no-wind-v3-tie-adjusted"
            )
            if snapshot_exists(snapshot_id, args.database_url):
                print(f"{snapshot_id}: already stored")
                continue
            inserted = persist_predictions(
                week_frame,
                snapshot_id=snapshot_id,
                model_version=(
                    f"rolling-through-{season - 1}-no-wind-v3-tie-adjusted"
                ),
                database_url=args.database_url,
                enforce_kickoff_lock=False,
            )
            print(f"{snapshot_id}: stored {inserted} predictions")


if __name__ == "__main__":
    main()
