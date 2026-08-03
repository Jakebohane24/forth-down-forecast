"""Load eligible 2026 schedule-only cards for the public interface."""

from __future__ import annotations

import argparse

import nfl_data_py as nfl
import pandas as pd

from api.ingest import persist_scheduled_games
from src.weather import TEAM_VENUES


NEUTRAL_VENUES = {
    "2026_01_SF_LA": ("Melbourne Cricket Ground", "outdoor", "AU"),
    "2026_03_BAL_DAL": ("Maracanã Stadium", "outdoor", "BR"),
    "2026_04_IND_WAS": ("Tottenham Hotspur Stadium", "outdoor", "GB"),
    "2026_06_HOU_JAX": ("Wembley Stadium", "outdoor", "GB"),
    "2026_07_PIT_NO": ("Stade de France", "outdoor", "FR"),
    "2026_09_CIN_ATL": ("Santiago Bernabéu Stadium", "outdoor", "ES"),
    "2026_10_NE_DET": ("Allianz Arena", "outdoor", "DE"),
    "2026_11_MIN_SF": ("Estadio Banorte", "outdoor", "MX"),
}


def normalized_schedule(season: int = 2026) -> pd.DataFrame:
    schedule = nfl.import_schedules([season])
    games = schedule.loc[schedule["game_type"] == "REG"].copy()
    local = pd.to_datetime(
        games["gameday"].astype(str)
        + " "
        + games["gametime"].fillna("00:00"),
        errors="coerce",
    )
    games["kickoff"] = local.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert("UTC")

    previous_games: dict[str, int] = {}
    eligibility = {}
    for week in sorted(games["week"].unique()):
        week_games = games.loc[games["week"] == week]
        for _, game in week_games.iterrows():
            eligibility[game["game_id"]] = (
                previous_games.get(game["home_team"], 0) >= 5
                and previous_games.get(game["away_team"], 0) >= 5
            )
        for team in pd.concat(
            [week_games["home_team"], week_games["away_team"]]
        ):
            previous_games[team] = previous_games.get(team, 0) + 1
    games["prediction_eligible"] = games["game_id"].map(eligibility)

    def venue(game):
        neutral = NEUTRAL_VENUES.get(game["game_id"])
        if neutral:
            return neutral
        home = TEAM_VENUES[game["home_team"]]
        venue_type = home.venue_type
        roof_status = (
            "closed"
            if venue_type == "indoor"
            else "pending"
            if venue_type == "retractable"
            else "open"
        )
        return home.name, venue_type, "US", roof_status

    venue_rows = []
    for _, game in games.iterrows():
        values = venue(game)
        if len(values) == 3:
            name, venue_type, country = values
            roof_status = "open"
        else:
            name, venue_type, country, roof_status = values
        venue_rows.append((name, venue_type, roof_status, country))
    venue_frame = pd.DataFrame(
        venue_rows,
        columns=["venue_name", "venue_type", "roof_status", "country_code"],
        index=games.index,
    )
    games[venue_frame.columns] = venue_frame
    return games[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "kickoff",
            "venue_name",
            "venue_type",
            "roof_status",
            "country_code",
            "prediction_eligible",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--database-url")
    args = parser.parse_args()
    games = normalized_schedule(args.season)
    count = persist_scheduled_games(games, database_url=args.database_url)
    eligible = int(games["prediction_eligible"].sum())
    print(f"Stored {count} scheduled games; {eligible} are prediction-card eligible")


if __name__ == "__main__":
    main()
