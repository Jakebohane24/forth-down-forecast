"""Schedule-provider normalization for upcoming NFL games."""

from __future__ import annotations

import nfl_data_py as nfl
import pandas as pd


def nflverse_week(season: int, week: int) -> pd.DataFrame:
    """Download and normalize one regular-season week from nflverse."""
    schedule = nfl.import_schedules([season])
    games = schedule.loc[
        (schedule["season"] == season)
        & (schedule["week"] == week)
        & (schedule["game_type"] == "REG")
    ].copy()
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "kickoff",
                "div_game",
                "game_wind",
                "stadium",
                "location",
            ]
        )

    # nflverse publishes `gametime` in US Eastern time. Localize before
    # converting so weather is selected for the actual kickoff hour in UTC.
    kickoff_eastern = pd.to_datetime(
        games["gameday"].astype(str) + " " + games["gametime"].fillna("00:00"),
        errors="coerce",
    )
    games["kickoff"] = kickoff_eastern.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")
    games["game_wind"] = pd.to_numeric(games.get("wind", 0), errors="coerce").fillna(0)
    games["div_game"] = pd.to_numeric(
        games.get("div_game", 0), errors="coerce"
    ).fillna(0).astype(int)
    return games[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "kickoff",
            "div_game",
            "game_wind",
            "stadium",
            "location",
        ]
    ].reset_index(drop=True)
