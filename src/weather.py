"""Pregame stadium weather from a reproducible, server-side provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import httpx
import pandas as pd


VenueType = Literal["outdoor", "indoor", "retractable"]


@dataclass(frozen=True)
class Venue:
    name: str
    latitude: float
    longitude: float
    venue_type: VenueType
    country_code: str = "US"


# Coordinates are stadium coordinates, not city centroids. Team lookup is the
# dependable default for ordinary home games; neutral-site games must provide
# explicit stadium_latitude/stadium_longitude values in the schedule.
TEAM_VENUES: dict[str, Venue] = {
    "ARI": Venue("State Farm Stadium", 33.5276, -112.2626, "retractable"),
    "ATL": Venue("Mercedes-Benz Stadium", 33.7554, -84.4008, "retractable"),
    "BAL": Venue("M&T Bank Stadium", 39.2780, -76.6227, "outdoor"),
    "BUF": Venue("Highmark Stadium", 42.7738, -78.7868, "outdoor"),
    "CAR": Venue("Bank of America Stadium", 35.2258, -80.8528, "outdoor"),
    "CHI": Venue("Soldier Field", 41.8623, -87.6167, "outdoor"),
    "CIN": Venue("Paycor Stadium", 39.0954, -84.5160, "outdoor"),
    "CLE": Venue("Huntington Bank Field", 41.5061, -81.6995, "outdoor"),
    "DAL": Venue("AT&T Stadium", 32.7473, -97.0945, "retractable"),
    "DEN": Venue("Empower Field at Mile High", 39.7439, -105.0201, "outdoor"),
    "DET": Venue("Ford Field", 42.3400, -83.0456, "indoor"),
    "GB": Venue("Lambeau Field", 44.5013, -88.0622, "outdoor"),
    "HOU": Venue("NRG Stadium", 29.6847, -95.4107, "retractable"),
    "IND": Venue("Lucas Oil Stadium", 39.7601, -86.1639, "retractable"),
    "JAX": Venue("EverBank Stadium", 30.3239, -81.6373, "outdoor"),
    "KC": Venue("GEHA Field at Arrowhead Stadium", 39.0489, -94.4839, "outdoor"),
    "LA": Venue("SoFi Stadium", 33.9535, -118.3392, "indoor"),
    "LAC": Venue("SoFi Stadium", 33.9535, -118.3392, "indoor"),
    "LAR": Venue("SoFi Stadium", 33.9535, -118.3392, "indoor"),
    "LV": Venue("Allegiant Stadium", 36.0908, -115.1830, "indoor"),
    "MIA": Venue("Hard Rock Stadium", 25.9580, -80.2389, "outdoor"),
    "MIN": Venue("U.S. Bank Stadium", 44.9736, -93.2575, "indoor"),
    "NE": Venue("Gillette Stadium", 42.0909, -71.2643, "outdoor"),
    "NO": Venue("Caesars Superdome", 29.9511, -90.0812, "indoor"),
    "NYG": Venue("MetLife Stadium", 40.8135, -74.0745, "outdoor"),
    "NYJ": Venue("MetLife Stadium", 40.8135, -74.0745, "outdoor"),
    "PHI": Venue("Lincoln Financial Field", 39.9008, -75.1675, "outdoor"),
    "PIT": Venue("Acrisure Stadium", 40.4468, -80.0158, "outdoor"),
    "SEA": Venue("Lumen Field", 47.5952, -122.3316, "outdoor"),
    "SF": Venue("Levi's Stadium", 37.4030, -121.9700, "outdoor"),
    "TB": Venue("Raymond James Stadium", 27.9759, -82.5033, "outdoor"),
    "TEN": Venue("Nissan Stadium", 36.1665, -86.7713, "outdoor"),
    "WAS": Venue("Northwest Stadium", 38.9076, -76.8645, "outdoor"),
}


@dataclass(frozen=True)
class WeatherForecast:
    game_id: str
    venue_name: str
    venue_type: VenueType
    roof_status: str
    forecast_for: datetime
    retrieved_at: datetime
    wind_mph: float | None
    wind_gust_mph: float | None
    temperature_f: float | None
    precipitation_probability: float | None
    country_code: str = "US"
    precipitation_inches: float | None = None
    weather_code: int | None = None
    source: str = "open-meteo"

    @property
    def model_wind_mph(self) -> float:
        # A fixed indoor venue has no field-level wind. Until a retractable
        # roof is confirmed open, the locked forecast uses the indoor fallback
        # instead of pretending the public roof decision is known.
        if self.venue_type in {"indoor", "retractable"}:
            return 0.0
        if self.wind_mph is None:
            raise ValueError(f"No wind forecast available for outdoor game {self.game_id}")
        return self.wind_mph


class OpenMeteoWeatherProvider:
    """Fetch hourly kickoff conditions from Open-Meteo without an API key."""

    endpoint = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 15):
        self._client = client
        self.timeout = timeout

    def _request(self, latitude: float, longitude: float, kickoff: datetime) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "temperature_2m,precipitation_probability,precipitation,"
                "weather_code,"
                "wind_speed_10m,wind_gusts_10m"
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "start_date": kickoff.date().isoformat(),
            "end_date": kickoff.date().isoformat(),
        }
        if self._client is not None:
            response = self._client.get(self.endpoint, params=params)
        else:
            response = httpx.get(self.endpoint, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def forecast(self, game: pd.Series, *, retrieved_at: datetime | None = None) -> WeatherForecast:
        kickoff = pd.Timestamp(game.get("kickoff"))
        if pd.isna(kickoff):
            raise ValueError(f"{game['game_id']} has no valid kickoff time")
        if kickoff.tzinfo is None:
            kickoff = kickoff.tz_localize(UTC)
        else:
            kickoff = kickoff.tz_convert(UTC)

        neutral = str(game.get("location", "")).lower() == "neutral"
        has_override = pd.notna(game.get("stadium_latitude")) and pd.notna(
            game.get("stadium_longitude")
        )
        if neutral and not has_override:
            raise ValueError(
                f"{game['game_id']} is neutral-site; stadium coordinates are required"
            )

        venue = TEAM_VENUES.get(str(game["home_team"]))
        if venue is None and not has_override:
            raise ValueError(f"No stadium coordinates configured for {game['home_team']}")
        latitude = float(game["stadium_latitude"]) if has_override else venue.latitude
        longitude = float(game["stadium_longitude"]) if has_override else venue.longitude
        stadium = game.get("stadium")
        venue_name = (
            str(stadium)
            if pd.notna(stadium) and str(stadium).strip()
            else venue.name
            if venue
            else "Neutral site"
        )
        venue_type = str(game.get("venue_type") or (venue.venue_type if venue else "outdoor"))
        if venue_type not in {"outdoor", "indoor", "retractable"}:
            raise ValueError(f"Unsupported venue type {venue_type!r}")

        payload = self._request(latitude, longitude, kickoff.to_pydatetime())
        hourly = payload["hourly"]
        times = pd.to_datetime(hourly["time"], utc=True)
        if times.empty:
            raise ValueError(f"No hourly forecast returned for {game['game_id']}")
        position = int(abs(times - kickoff).argmin())

        def value(name: str) -> float | None:
            values = hourly.get(name)
            if values is None:
                return None
            raw = values[position]
            return None if raw is None else float(raw)

        roof_status = (
            "closed"
            if venue_type == "indoor"
            else "pending"
            if venue_type == "retractable"
            else "open"
        )
        return WeatherForecast(
            game_id=str(game["game_id"]),
            venue_name=venue_name,
            venue_type=venue_type,  # type: ignore[arg-type]
            roof_status=roof_status,
            forecast_for=times[position].to_pydatetime(),
            retrieved_at=retrieved_at or datetime.now(UTC),
            wind_mph=value("wind_speed_10m"),
            wind_gust_mph=value("wind_gusts_10m"),
            temperature_f=value("temperature_2m"),
            precipitation_probability=value("precipitation_probability"),
            country_code=(
                str(game.get("country_code"))
                if pd.notna(game.get("country_code"))
                else venue.country_code
                if venue
                else "US"
            ),
            precipitation_inches=value("precipitation"),
            weather_code=int(value("weather_code"))
            if value("weather_code") is not None
            else None,
        )


def enrich_schedule_with_weather(
    schedule: pd.DataFrame,
    provider: OpenMeteoWeatherProvider | None = None,
) -> tuple[pd.DataFrame, list[WeatherForecast]]:
    """Return a copy with leakage-safe model wind plus auditable conditions."""
    provider = provider or OpenMeteoWeatherProvider()
    enriched = schedule.copy()
    forecasts = [provider.forecast(game) for _, game in enriched.iterrows()]
    wind_by_game = {forecast.game_id: forecast.model_wind_mph for forecast in forecasts}
    enriched["game_wind"] = enriched["game_id"].map(wind_by_game)
    return enriched, forecasts
