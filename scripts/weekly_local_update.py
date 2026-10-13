"""Refresh local NFL data and generate the next eligible weekly forecast."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processing import build_features


def run_module(module: str, *arguments: str) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def next_upcoming_week(season: int) -> int:
    schedule = nfl.import_schedules([season])
    schedule = schedule.loc[schedule["game_type"].eq("REG")].copy()
    kickoff = pd.to_datetime(
        schedule["gameday"].astype(str) + " " + schedule["gametime"].fillna("00:00"),
        errors="coerce",
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    upcoming = schedule.loc[kickoff > pd.Timestamp.now(tz="UTC")]
    if upcoming.empty:
        raise RuntimeError(f"No upcoming regular-season games found for {season}")
    return int(upcoming["week"].min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=datetime.now().year)
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--odds-api-key", default=None)
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    season = args.season
    print(f"Refreshing local forecast data for {season}", flush=True)

    # Rebuild the local raw play and feature tables through the current season.
    build_features(download=True, seasons=range(2015, season + 1))

    database_arguments = (["--database-url", args.database_url] if args.database_url else [])
    run_module("scripts.sync_results", "--season", str(season), *database_arguments)
    run_module("scripts.load_upcoming_schedule", *database_arguments)

    week = next_upcoming_week(season)
    arguments = [str(season), str(week)]
    if args.skip_weather:
        arguments.append("--skip-weather")
    if args.odds_api_key:
        arguments.extend(["--odds-api-key", args.odds_api_key])
    if args.database_url:
        arguments.extend(["--database-url", args.database_url])
    run_module("scripts.generate_week", *arguments)
    print(f"Local update complete: generated eligible predictions for week {week}", flush=True)


if __name__ == "__main__":
    main()
