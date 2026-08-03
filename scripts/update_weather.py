"""Refresh website weather without changing a locked prediction snapshot."""

from __future__ import annotations

import argparse

from api.ingest import persist_game_conditions
from src.schedule import nflverse_week
from src.weather import enrich_schedule_with_weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("week", type=int)
    parser.add_argument("--database-url")
    args = parser.parse_args()

    schedule = nflverse_week(args.season, args.week)
    if schedule.empty:
        raise RuntimeError(
            f"No regular-season games found for {args.season} week {args.week}"
        )
    _, forecasts = enrich_schedule_with_weather(schedule)
    count = persist_game_conditions(forecasts, database_url=args.database_url)
    print(f"Updated kickoff weather for {count} games; predictions were unchanged")


if __name__ == "__main__":
    main()
