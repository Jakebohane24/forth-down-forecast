from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest

from src.weather import OpenMeteoWeatherProvider, enrich_schedule_with_weather


def _client():
    def handler(request: httpx.Request):
        assert request.url.params["wind_speed_unit"] == "mph"
        assert request.url.params["timezone"] == "UTC"
        return httpx.Response(
            200,
            json={
                "hourly": {
                    "time": [
                        "2026-10-04T16:00",
                        "2026-10-04T17:00",
                        "2026-10-04T18:00",
                    ],
                    "temperature_2m": [58.0, 60.0, 61.0],
                    "precipitation_probability": [10, 20, 30],
                    "wind_speed_10m": [11.0, 13.5, 14.0],
                    "wind_gusts_10m": [18.0, 21.0, 23.0],
                }
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_outdoor_forecast_becomes_model_wind():
    schedule = pd.DataFrame(
        {
            "game_id": ["2026_05_MIA_BUF"],
            "season": [2026],
            "week": [5],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "kickoff": [datetime(2026, 10, 4, 17, 20, tzinfo=UTC)],
            "location": ["Home"],
        }
    )
    provider = OpenMeteoWeatherProvider(client=_client())

    enriched, forecasts = enrich_schedule_with_weather(schedule, provider)

    assert enriched.loc[0, "game_wind"] == 13.5
    assert forecasts[0].roof_status == "open"
    assert forecasts[0].forecast_for == datetime(2026, 10, 4, 17, tzinfo=UTC)


def test_retractable_roof_uses_zero_until_status_is_confirmed():
    game = pd.Series(
        {
            "game_id": "2026_05_NYG_DAL",
            "home_team": "DAL",
            "away_team": "NYG",
            "kickoff": datetime(2026, 10, 4, 17, 20, tzinfo=UTC),
            "location": "Home",
        }
    )
    forecast = OpenMeteoWeatherProvider(client=_client()).forecast(game)

    assert forecast.wind_mph == 13.5
    assert forecast.roof_status == "pending"
    assert forecast.model_wind_mph == 0


def test_neutral_site_requires_explicit_coordinates():
    game = pd.Series(
        {
            "game_id": "2026_05_TEAM_TEAM",
            "home_team": "JAX",
            "away_team": "CHI",
            "kickoff": datetime(2026, 10, 4, 17, tzinfo=UTC),
            "location": "Neutral",
        }
    )

    with pytest.raises(ValueError, match="stadium coordinates"):
        OpenMeteoWeatherProvider(client=_client()).forecast(game)
