"""Backfill real historical kickoff weather for public showcase games."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
import time

import httpx
import nfl_data_py as nfl
import pandas as pd

from api.ingest import persist_game_conditions
from src.weather import TEAM_VENUES, WeatherForecast


ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
CACHE_PATH = Path("artifacts/weather/historical_conditions_2022_2025.parquet")
SEASONS = range(2022, 2026)


# nflverse correctly marks these games neutral, but several 2025 stadium labels
# contain the designated home venue rather than the international venue.
NEUTRAL_VENUES = {
    "2022_08_DEN_JAX": ("Wembley Stadium", 51.5560, -0.2796, "outdoor", "GB"),
    "2022_10_SEA_TB": ("Allianz Arena", 48.2188, 11.6247, "outdoor", "DE"),
    "2022_11_CLE_BUF": ("Ford Field", 42.3400, -83.0456, "indoor", "US"),
    "2022_11_SF_ARI": ("Estadio Azteca", 19.3029, -99.1505, "outdoor", "MX"),
    "2023_06_BAL_TEN": ("Tottenham Hotspur Stadium", 51.6043, -0.0664, "outdoor", "GB"),
    "2023_09_MIA_KC": ("Deutsche Bank Park", 50.0686, 8.6455, "outdoor", "DE"),
    "2023_10_IND_NE": ("Deutsche Bank Park", 50.0686, 8.6455, "outdoor", "DE"),
    "2024_06_JAX_CHI": ("Tottenham Hotspur Stadium", 51.6043, -0.0664, "outdoor", "GB"),
    "2024_07_NE_JAX": ("Wembley Stadium", 51.5560, -0.2796, "outdoor", "GB"),
    "2024_10_NYG_CAR": ("Allianz Arena", 48.2188, 11.6247, "outdoor", "DE"),
    "2025_06_DEN_NYJ": ("Tottenham Hotspur Stadium", 51.6043, -0.0664, "outdoor", "GB"),
    "2025_07_LA_JAX": ("Wembley Stadium", 51.5560, -0.2796, "outdoor", "GB"),
    "2025_10_ATL_IND": ("Olympiastadion Berlin", 52.5147, 13.2395, "outdoor", "DE"),
    "2025_11_WAS_MIA": ("Santiago Bernabéu Stadium", 40.4531, -3.6883, "outdoor", "ES"),
}


def prepare_games() -> pd.DataFrame:
    schedule = nfl.import_schedules(list(SEASONS))
    games = schedule.loc[
        (schedule["game_type"] == "REG") & schedule["week"].between(6, 18)
    ].copy()
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
    ).dt.tz_convert(UTC)

    def venue_values(game: pd.Series) -> tuple[str, float, float, str, str]:
        override = NEUTRAL_VENUES.get(str(game["game_id"]))
        if override:
            return override
        venue = TEAM_VENUES[str(game["home_team"])]
        roof = str(game.get("roof", "")).lower()
        venue_type = "indoor" if roof in {"dome", "closed"} else "outdoor"
        return venue.name, venue.latitude, venue.longitude, venue_type, "US"

    values = games.apply(venue_values, axis=1, result_type="expand")
    values.columns = [
        "venue_name",
        "latitude",
        "longitude",
        "venue_type",
        "country_code",
    ]
    games[values.columns] = values
    return games


def indoor_forecasts(games: pd.DataFrame, retrieved_at: datetime) -> list[WeatherForecast]:
    return [
        WeatherForecast(
            game_id=str(game["game_id"]),
            venue_name=str(game["venue_name"]),
            venue_type="indoor",
            roof_status="closed",
            forecast_for=pd.Timestamp(game["kickoff"]).to_pydatetime(),
            retrieved_at=retrieved_at,
            wind_mph=0,
            wind_gust_mph=0,
            temperature_f=None,
            precipitation_probability=None,
            country_code=str(game["country_code"]),
            precipitation_inches=0,
            weather_code=None,
            source="open-meteo-historical-forecast",
        )
        for _, game in games.loc[games["venue_type"] == "indoor"].iterrows()
    ]


def fetch_date(games: pd.DataFrame, retrieved_at: datetime):
    date = games.iloc[0]["kickoff"].date().isoformat()
    params = {
        "latitude": ",".join(games["latitude"].astype(str)),
        "longitude": ",".join(games["longitude"].astype(str)),
        "hourly": (
            "temperature_2m,precipitation_probability,precipitation,"
            "weather_code,wind_speed_10m,wind_gusts_10m"
        ),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
        "start_date": date,
        "end_date": date,
    }
    response = None
    for attempt in range(6):
        response = httpx.get(
            ENDPOINT,
            params=params,
            timeout=60,
            follow_redirects=True,
        )
        if response.status_code != 429:
            break
        retry_after = response.headers.get("Retry-After")
        wait_seconds = (
            float(retry_after)
            if retry_after and retry_after.isdigit()
            else min(5 * 2**attempt, 40)
        )
        time.sleep(wait_seconds)
    if response is None:
        raise RuntimeError(f"No weather response for {date}")
    response.raise_for_status()
    payload = response.json()
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(games):
        raise RuntimeError(f"Weather location count mismatch for {date}")

    forecasts = []
    for (_, game), weather in zip(games.iterrows(), payloads, strict=True):
        hourly = weather["hourly"]
        times = pd.to_datetime(hourly["time"], utc=True)
        position = int(abs(times - game["kickoff"]).argmin())

        def value(name):
            values = hourly.get(name)
            raw = values[position] if values is not None else None
            return None if raw is None else float(raw)

        code = value("weather_code")
        forecasts.append(
            WeatherForecast(
                game_id=str(game["game_id"]),
                venue_name=str(game["venue_name"]),
                venue_type="outdoor",
                roof_status="open",
                forecast_for=times[position].to_pydatetime(),
                retrieved_at=retrieved_at,
                wind_mph=value("wind_speed_10m"),
                wind_gust_mph=value("wind_gusts_10m"),
                temperature_f=value("temperature_2m"),
                precipitation_probability=value("precipitation_probability"),
                country_code=str(game["country_code"]),
                precipitation_inches=value("precipitation"),
                weather_code=int(code) if code is not None else None,
                source="open-meteo-historical-forecast",
            )
        )
    return forecasts


def download_conditions(games: pd.DataFrame) -> list[WeatherForecast]:
    retrieved_at = datetime.now(UTC)
    forecasts = indoor_forecasts(games, retrieved_at)
    outdoor = games.loc[games["venue_type"] == "outdoor"]
    grouped: dict[object, list[int]] = defaultdict(list)
    for index, kickoff in outdoor["kickoff"].items():
        grouped[kickoff.date()].append(index)
    batches = [outdoor.loc[indices] for _, indices in sorted(grouped.items())]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(fetch_date, batch, retrieved_at): batch
            for batch in batches
        }
        for number, future in enumerate(as_completed(futures), start=1):
            forecasts.extend(future.result())
            if number % 20 == 0 or number == len(batches):
                print(
                    f"Downloaded {number}/{len(batches)} game dates",
                    flush=True,
                )
    return forecasts


def save_cache(forecasts: list[WeatherForecast]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([forecast.__dict__ for forecast in forecasts]).to_parquet(
        CACHE_PATH, index=False
    )


def load_cache() -> list[WeatherForecast]:
    rows = pd.read_parquet(CACHE_PATH).to_dict(orient="records")
    return [
        WeatherForecast(
            **{
                **row,
                "forecast_for": pd.Timestamp(row["forecast_for"]).to_pydatetime(),
                "retrieved_at": pd.Timestamp(row["retrieved_at"]).to_pydatetime(),
                "weather_code": int(row["weather_code"])
                if pd.notna(row["weather_code"])
                else None,
                "country_code": NEUTRAL_VENUES.get(
                    str(row["game_id"]),
                    (None, None, None, None, "US"),
                )[4],
            }
        )
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-url")
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()
    if CACHE_PATH.exists() and not args.refresh_cache:
        forecasts = load_cache()
    else:
        forecasts = download_conditions(prepare_games())
        save_cache(forecasts)
    count = persist_game_conditions(forecasts, database_url=args.database_url)
    print(f"Stored historical kickoff conditions for {count} games")


if __name__ == "__main__":
    main()
