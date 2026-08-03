"""Test frozen rolling models with only test-season wind replaced.

The replacement is Open-Meteo's stitched historical forecast at the home
stadium one hour before kickoff. Training rows and fitted models are untouched.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC
from pathlib import Path

import httpx
import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.rolling_backtest import moneyline_results, pooled_results
from src.config import BettingConfig, EvaluationConfig
from src.evaluation import evaluation_frame
from src.training import NFLModel
from src.weather import TEAM_VENUES


ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
SEASONS = range(2022, 2026)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = PROJECT_ROOT / "artifacts" / "weather" / "test_wind_2022_2025.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "test_wind_replacement_2022_2025.json"


def _kickoff_utc(schedule: pd.DataFrame) -> pd.Series:
    local = pd.to_datetime(
        schedule["gameday"].astype(str)
        + " "
        + schedule["gametime"].fillna("00:00"),
        errors="coerce",
    )
    return local.dt.tz_localize(
        "America/New_York",
        ambiguous="NaT",
        nonexistent="shift_forward",
    ).dt.tz_convert(UTC)


def _eligible_games(schedule: pd.DataFrame, model_games: set[str]) -> pd.DataFrame:
    games = schedule.loc[
        schedule["game_id"].isin(model_games)
        & (schedule["game_type"] == "REG")
    ].copy()
    games["kickoff"] = _kickoff_utc(games)
    games["weather_time"] = games["kickoff"] - pd.Timedelta(hours=1)
    games["venue"] = games["home_team"].map(TEAM_VENUES)
    games["latitude"] = games["venue"].map(
        lambda venue: venue.latitude if venue else np.nan
    )
    games["longitude"] = games["venue"].map(
        lambda venue: venue.longitude if venue else np.nan
    )
    games["neutral_site"] = games["location"].str.lower().eq("neutral")
    return games


def _fetch_date(
    client: httpx.Client,
    games: pd.DataFrame,
) -> list[dict]:
    params = {
        "latitude": ",".join(games["latitude"].astype(str)),
        "longitude": ",".join(games["longitude"].astype(str)),
        "hourly": "wind_speed_10m",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "start_date": games.iloc[0]["weather_time"].date().isoformat(),
        "end_date": games.iloc[0]["weather_time"].date().isoformat(),
    }
    response = client.get(ENDPOINT, params=params)
    response.raise_for_status()
    payload = response.json()
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(games):
        raise RuntimeError(
            f"Expected {len(games)} weather locations, received {len(payloads)}"
        )

    rows = []
    for (_, game), weather in zip(games.iterrows(), payloads, strict=True):
        times = pd.to_datetime(weather["hourly"]["time"], utc=True)
        position = int(abs(times - game["weather_time"]).argmin())
        rows.append(
            {
                "game_id": game["game_id"],
                "season": int(game["season"]),
                "week": int(game["week"]),
                "weather_time": game["weather_time"],
                "weather_grid_time": times[position],
                "historical_wind_mph": float(
                    weather["hourly"]["wind_speed_10m"][position]
                ),
                "latitude": float(game["latitude"]),
                "longitude": float(game["longitude"]),
                "weather_source": "open-meteo-historical-forecast",
            }
        )
    return rows


def download_wind(schedule: pd.DataFrame, model_games: set[str]) -> pd.DataFrame:
    games = _eligible_games(schedule, model_games)
    regular_home = games.loc[
        ~games["neutral_site"] & games["latitude"].notna() & games["kickoff"].notna()
    ].copy()
    rows: list[dict] = []
    grouped: dict[object, list[int]] = defaultdict(list)
    for index, value in regular_home["weather_time"].items():
        grouped[value.date()].append(index)

    with httpx.Client(timeout=45, follow_redirects=True) as client:
        for number, (_, indices) in enumerate(sorted(grouped.items()), start=1):
            day_games = regular_home.loc[indices]
            rows.extend(_fetch_date(client, day_games))
            if number % 20 == 0:
                print(f"Downloaded {number}/{len(grouped)} game dates")

    wind = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    wind.to_parquet(CACHE_PATH, index=False)
    excluded = games.loc[~games["game_id"].isin(wind["game_id"])]
    print(
        f"Cached {len(wind)} games; excluded {len(excluded)} "
        "neutral/missing-location games"
    )
    return wind


def _metrics(frame: pd.DataFrame) -> dict:
    return {
        "games": len(frame),
        "margin_mae": float(
            mean_absolute_error(frame["actual_margin"], frame["pred_spread"])
        ),
        "margin_rmse": float(
            mean_squared_error(frame["actual_margin"], frame["pred_spread"]) ** 0.5
        ),
        "win_accuracy": float(
            (frame["pred_home_win"] == frame["actual_home_win"]).mean()
        ),
    }


def _evaluate_with_wind(
    model: NFLModel,
    replacements: pd.Series,
    schedules: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    mask = model.df["season"] == model.config.test_season
    original = model.df.loc[mask, "game_wind"].copy()
    # Preserve the established indoor/closed-roof zero after replacing only
    # outdoor rows. This isolates wind rather than turning roof into a new test.
    external = replacements.reindex(original.index)
    usable = external.notna() & model.df.loc[mask, "roof_adjusted"].eq(0)
    model.df.loc[original.index[usable], "game_wind"] = external.loc[usable]
    try:
        frame = evaluation_frame(model, "test", config=EvaluationConfig())
    finally:
        model.df.loc[original.index, "game_wind"] = original
    betting = moneyline_results(
        frame,
        schedules,
        BettingConfig().moneyline_confidence_threshold,
        BettingConfig().moneyline_minimum_odds,
    )
    result = {
        **_metrics(frame),
        "wind_replaced_games": int(usable.sum()),
        "wind_unchanged_games": int((~usable).sum()),
        "moneyline_signal": betting,
    }
    return frame, result


def main() -> None:
    schedules = nfl.import_schedules(list(SEASONS))
    models = {
        season: NFLModel.load(PROJECT_ROOT / "artifacts" / "models" / f"rolling_{season}")
        for season in SEASONS
    }
    model_games = {
        game_id
        for model in models.values()
        for game_id in model.df.index[model.df["season"] == model.config.test_season]
    }
    wind = (
        pd.read_parquet(CACHE_PATH)
        if CACHE_PATH.exists()
        else download_wind(schedules, model_games)
    )
    replacements = wind.set_index("game_id")["historical_wind_mph"]

    seasons = []
    baseline_seasons = []
    paired_changes = []
    for season, model in models.items():
        baseline = evaluation_frame(model, "test", config=EvaluationConfig())
        replacement_frame, replacement = _evaluate_with_wind(
            model, replacements, schedules
        )
        baseline_result = {
            "test_season": season,
            **_metrics(baseline),
            "moneyline_signal": moneyline_results(
                baseline,
                schedules,
                BettingConfig().moneyline_confidence_threshold,
                BettingConfig().moneyline_minimum_odds,
            ),
        }
        replacement["test_season"] = season
        baseline_seasons.append(baseline_result)
        seasons.append(replacement)
        paired_changes.append(
            pd.DataFrame(
                {
                    "actual_margin": baseline["actual_margin"],
                    "baseline_spread": baseline["pred_spread"],
                    "replacement_spread": replacement_frame["pred_spread"],
                    "baseline_home_win": baseline["pred_home_win"],
                    "replacement_home_win": replacement_frame["pred_home_win"],
                }
            )
        )
        print(
            f"{season}: MAE {baseline_result['margin_mae']:.3f} -> "
            f"{replacement['margin_mae']:.3f}; win "
            f"{baseline_result['win_accuracy']:.2%} -> "
            f"{replacement['win_accuracy']:.2%}; signal ROI "
            f"{baseline_result['moneyline_signal']['roi']:.2%} -> "
            f"{replacement['moneyline_signal']['roi']:.2%}"
        )

    paired = pd.concat(paired_changes)
    report = {
        "description": (
            "Frozen expanding-window models with training data untouched. Only "
            "test-season outdoor game_wind was replaced by Open-Meteo's "
            "stadium-location historical forecast one hour before kickoff."
        ),
        "seasons": list(SEASONS),
        "weather_source": ENDPOINT,
        "weather_cache": str(CACHE_PATH.relative_to(PROJECT_ROOT)),
        "neutral_site_policy": "Original wind retained; no home-stadium substitution.",
        "roof_policy": "Original zero retained when historical roof_adjusted=1.",
        "baseline": {
            "pooled_moneyline_signal": pooled_results(
                baseline_seasons, "moneyline_signal"
            ),
            "by_season": baseline_seasons,
        },
        "replacement": {
            "pooled_moneyline_signal": pooled_results(seasons, "moneyline_signal"),
            "by_season": seasons,
        },
        "prediction_changes": {
            "games": len(paired),
            "spread_changed": int(
                (paired["baseline_spread"] != paired["replacement_spread"]).sum()
            ),
            "winner_changed": int(
                (
                    paired["baseline_home_win"]
                    != paired["replacement_home_win"]
                ).sum()
            ),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
