"""Correct stored historical kickoff timestamps from the nflverse schedule."""

from __future__ import annotations

import argparse

import nfl_data_py as nfl
import pandas as pd
from sqlalchemy import select, update

from api.database import create_database
from api.models import Prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    args = parser.parse_args()
    engine, session_factory = create_database(args.database_url)
    with session_factory() as session:
        seasons = sorted(set(session.scalars(select(Prediction.season))))
    schedule = nfl.import_schedules(seasons)
    local = pd.to_datetime(
        schedule["gameday"].astype(str)
        + " "
        + schedule["gametime"].fillna("00:00"),
        errors="coerce",
    )
    schedule["kickoff"] = local.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")
    kickoff_by_game = schedule.set_index("game_id")["kickoff"].dropna()
    updated = 0
    with session_factory.begin() as session:
        for game_id, kickoff in kickoff_by_game.items():
            result = session.execute(
                update(Prediction)
                .where(Prediction.game_id == str(game_id))
                .values(kickoff=kickoff.to_pydatetime())
            )
            updated += result.rowcount
    engine.dispose()
    print(f"Corrected kickoff timestamps on {updated} prediction rows")


if __name__ == "__main__":
    main()
