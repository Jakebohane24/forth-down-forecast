"""Estimate uncertainty around the no-wind model's historical signal ROI."""

from __future__ import annotations

import json
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from scipy.stats import norm, t

from src.config import BettingConfig, ModelConfig
from src.evaluation import evaluation_frame
from src.training import NFLModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models" / "no_wind_rolling"
RETURNS_PATH = PROJECT_ROOT / "artifacts" / "evaluation" / "no_wind_signals.parquet"
REPORT_PATH = PROJECT_ROOT / "reports" / "roi_uncertainty_no_wind.json"
SEASONS = range(2022, 2026)
BOOTSTRAP_SAMPLES = 100_000
RANDOM_STATE = 24


def model_for_season(season: int) -> NFLModel:
    artifact = ARTIFACT_ROOT / str(season)
    if (artifact / "models.joblib").exists():
        print(f"{season}: loading cached no-wind model")
        return NFLModel.load(artifact)
    config = ModelConfig(
        use_wind=False,
        training_seasons=tuple(range(2018, season)),
        validation_season=season - 1,
        test_season=season,
    )
    print(f"{season}: training frozen no-wind expanding window")
    model = NFLModel(config).train()
    model.save(
        artifact,
        model_version=f"rolling-through-{season - 1}-no-wind-v1",
    )
    return model


def signal_returns(model: NFLModel, schedules: pd.DataFrame) -> pd.DataFrame:
    frame = evaluation_frame(model, "test")
    frame = frame.merge(
        schedules[
            [
                "season",
                "week",
                "home_team",
                "away_team",
                "home_moneyline",
                "away_moneyline",
            ]
        ],
        on=["season", "week", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    frame["confidence"] = frame["home_win_prob"].where(
        frame["pred_home_win"], 1 - frame["home_win_prob"]
    )
    frame["selected_odds"] = frame["home_moneyline"].where(
        frame["pred_home_win"], frame["away_moneyline"]
    )
    config = BettingConfig()
    signals = frame.loc[
        (frame["confidence"] >= config.moneyline_confidence_threshold)
        & frame["selected_odds"].notna()
        & (frame["selected_odds"] >= config.moneyline_minimum_odds)
    ].copy()
    signals["won"] = (
        signals["pred_home_win"]
        & (signals["home_score"] > signals["away_score"])
    ) | (
        ~signals["pred_home_win"]
        & (signals["away_score"] > signals["home_score"])
    )
    winning_return = signals["selected_odds"].where(
        signals["selected_odds"] > 0,
        100,
    ).div(100).where(
        signals["selected_odds"] > 0,
        100 / signals["selected_odds"].abs(),
    )
    signals["return_units"] = winning_return.where(signals["won"], -1.0)
    return signals[
        [
            "season",
            "week",
            "home_team",
            "away_team",
            "selected_odds",
            "confidence",
            "won",
            "return_units",
        ]
    ]


def wilson_interval(wins: int, count: int, confidence: float = 0.95) -> list[float]:
    """Wilson score interval for a binomial win proportion."""
    observed = wins / count
    z = float(norm.ppf((1 + confidence) / 2))
    denominator = 1 + z**2 / count
    center = (observed + z**2 / (2 * count)) / denominator
    half_width = (
        z
        * np.sqrt(
            observed * (1 - observed) / count
            + z**2 / (4 * count**2)
        )
        / denominator
    )
    return [center - half_width, center + half_width]


def interval_report(returns: np.ndarray, wins: int) -> dict:
    count = len(returns)
    mean = float(returns.mean())
    standard_deviation = float(returns.std(ddof=1))
    standard_error = standard_deviation / np.sqrt(count)
    t_critical = float(t.ppf(0.975, df=count - 1))
    t_interval = [
        mean - t_critical * standard_error,
        mean + t_critical * standard_error,
    ]

    rng = np.random.default_rng(RANDOM_STATE)
    bootstrap_means = rng.choice(
        returns,
        size=(BOOTSTRAP_SAMPLES, count),
        replace=True,
    ).mean(axis=1)
    bootstrap_se = float(bootstrap_means.std(ddof=1))
    percentile_interval = np.quantile(
        bootstrap_means, [0.025, 0.975]
    ).tolist()
    normal_bootstrap_interval = [
        mean - 1.96 * bootstrap_se,
        mean + 1.96 * bootstrap_se,
    ]
    # For a central two-sided interval, this is the confidence-level boundary
    # where the lower endpoint reaches zero. Intervals strictly below the
    # boundary exclude zero; intervals at it touch zero.
    t_positive_boundary = float(
        2 * t.cdf(mean / standard_error, df=count - 1) - 1
    )
    normal_positive_boundary = float(
        2 * norm.cdf(mean / bootstrap_se) - 1
    )
    percentile_positive_boundary = float(
        1 - 2 * np.mean(bootstrap_means <= 0)
    )
    return {
        "signals": count,
        "wins": wins,
        "losses": count - wins,
        "win_accuracy": wins / count,
        "win_accuracy_wilson_95": wilson_interval(wins, count),
        "mean_roi": mean,
        "profit_units": float(returns.sum()),
        "return_standard_deviation": standard_deviation,
        "mean_standard_error": standard_error,
        "confidence_level": 0.95,
        "student_t_interval": t_interval,
        "student_t_positive_roi_confidence_boundary": t_positive_boundary,
        "bootstrap": {
            "method": "ordinary nonparametric bootstrap of individual signals",
            "samples": BOOTSTRAP_SAMPLES,
            "random_state": RANDOM_STATE,
            "standard_error": bootstrap_se,
            "normal_interval": normal_bootstrap_interval,
            "percentile_interval": percentile_interval,
            "normal_positive_roi_confidence_boundary": (
                normal_positive_boundary
            ),
            "percentile_positive_roi_confidence_boundary": (
                percentile_positive_boundary
            ),
        },
    }


def main() -> None:
    schedules = nfl.import_schedules(list(SEASONS))
    signals = pd.concat(
        [signal_returns(model_for_season(season), schedules) for season in SEASONS],
        ignore_index=True,
    )
    RETURNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(RETURNS_PATH, index=False)
    report = {
        "description": (
            "Uncertainty intervals for the historical mean one-unit return of "
            "the locked 65% confidence / -300 moneyline rule using no-wind, "
            "split-tie probabilities and "
            "expanding-window test predictions from 2022 through 2025."
        ),
        "selection_caveat": (
            "Intervals are conditional on the retrospectively selected signal "
            "rule and do not account for threshold-selection bias or guarantee "
            "future profitability."
        ),
        "independence_caveat": (
            "The individual-signal intervals do not model correlation within "
            "weeks or seasons."
        ),
        **interval_report(
            signals["return_units"].to_numpy(),
            int(signals["won"].sum()),
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved signal returns to {RETURNS_PATH}")
    print(f"Saved interval report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
