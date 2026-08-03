"""Import completed production-season results from nflverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from api.ingest import persist_game_results


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = PROJECT_ROOT / "reports" / "production_model.json"


def production_season() -> int:
    metadata = json.loads(MODEL_MANIFEST.read_text())
    return max(metadata["training_seasons"]) + 1


def completed_games(season: int, week: int | None = None) -> pd.DataFrame:
    schedule = nfl.import_schedules([season])
    mask = (
        (schedule["season"] == season)
        & (schedule["game_type"] == "REG")
        & schedule["home_score"].notna()
        & schedule["away_score"].notna()
    )
    if week is not None:
        mask &= schedule["week"] == week
    games = schedule.loc[
        mask,
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
    ].copy()
    if games.empty:
        return games.set_index("game_id")
    games["home_score"] = pd.to_numeric(games["home_score"]).astype(int)
    games["away_score"] = pd.to_numeric(games["away_score"]).astype(int)
    return games.set_index("game_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=production_season())
    parser.add_argument("--week", type=int)
    parser.add_argument("--database-url")
    args = parser.parse_args()

    games = completed_games(args.season, args.week)
    if games.empty:
        print("No new completed regular-season games are available")
        return
    count = persist_game_results(games, database_url=args.database_url)
    print(f"Stored or refreshed {count} completed {args.season} games")


if __name__ == "__main__":
    main()
