"""Evaluate total-points forecasts against nflverse closing totals."""

from __future__ import annotations

import json
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.evaluation import evaluation_frame
from src.training import NFLModel


TEST_SEASONS = [2022, 2023, 2024, 2025]
EDGE_THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
ARTIFACT_ROOT = Path("artifacts/models")
OUTPUT_PATH = Path("reports/totals_backtest.json")


def total_forecast_metrics(frame: pd.DataFrame) -> dict:
    actual = frame["actual_total"]
    predicted = frame["pred_over_under"]
    error = predicted - actual
    return {
        "games": len(frame),
        "total_mae": float(mean_absolute_error(actual, predicted)),
        "total_rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "mean_bias_predicted_minus_actual": float(error.mean()),
        "correlation": float(predicted.corr(actual)),
        "within_3_points": float((error.abs() <= 3).mean()),
        "within_7_points": float((error.abs() <= 7).mean()),
        "within_10_points": float((error.abs() <= 10).mean()),
        "market_total_mae": float(
            mean_absolute_error(actual, frame["total_line"])
        ),
    }


def betting_results(frame: pd.DataFrame, threshold: float) -> dict:
    eligible = frame.loc[
        frame["total_line"].notna()
        & (~frame["is_total_push"])
        & (frame["totals_edge"].abs() > 0)
        & (frame["totals_edge"].abs() >= threshold)
    ].copy()
    eligible["won"] = eligible["model_over"] == eligible["actual_over"]
    eligible["profit"] = eligible["won"].map({True: 100 / 110, False: -1.0})

    def side(side_name: str) -> dict:
        selected = eligible.loc[eligible["model_side"] == side_name]
        return {
            "bets": len(selected),
            "wins": int(selected["won"].sum()),
            "accuracy": (
                float(selected["won"].mean()) if not selected.empty else None
            ),
            "profit_units": float(selected["profit"].sum()),
            "roi": (
                float(selected["profit"].mean()) if not selected.empty else None
            ),
        }

    return {
        "edge_threshold": threshold,
        "bets": len(eligible),
        "wins": int(eligible["won"].sum()),
        "losses": int((~eligible["won"]).sum()),
        "accuracy": (
            float(eligible["won"].mean()) if not eligible.empty else None
        ),
        "profit_units": float(eligible["profit"].sum()),
        "roi": float(eligible["profit"].mean()) if not eligible.empty else None,
        "overs": side("over"),
        "unders": side("under"),
    }


def prepare_frame(model: NFLModel, season: int, totals: pd.DataFrame) -> pd.DataFrame:
    frame = evaluation_frame(model, "test").join(totals, how="left")
    if frame["total_line"].isna().any():
        missing = int(frame["total_line"].isna().sum())
        raise RuntimeError(f"{season} is missing closing totals for {missing} games")
    frame["actual_total"] = frame["home_score"] + frame["away_score"]
    frame["totals_edge"] = frame["pred_over_under"] - frame["total_line"]
    frame["model_over"] = frame["totals_edge"] > 0
    frame["model_side"] = np.where(frame["model_over"], "over", "under")
    frame["actual_over"] = frame["actual_total"] > frame["total_line"]
    frame["is_total_push"] = frame["actual_total"] == frame["total_line"]
    frame["test_season"] = season
    return frame


def group_summary(frames: list[pd.DataFrame], seasons: list[int]) -> dict:
    pooled = pd.concat(
        [frame for frame in frames if int(frame["test_season"].iloc[0]) in seasons]
    )
    return {
        "seasons": seasons,
        "forecast": total_forecast_metrics(pooled),
        "thresholds": {
            str(threshold): betting_results(pooled, threshold)
            for threshold in EDGE_THRESHOLDS
        },
    }


def main() -> None:
    schedule = nfl.import_schedules(TEST_SEASONS)
    totals = schedule.set_index("game_id")[["total_line"]]
    frames = []
    season_results = []

    for season in TEST_SEASONS:
        artifact = ARTIFACT_ROOT / f"rolling_{season}"
        if not (artifact / "models.joblib").exists():
            raise RuntimeError(
                f"Missing cached rolling model for {season}. Run "
                "python -m scripts.build_historical_showcase first."
            )
        model = NFLModel.load(artifact)
        frame = prepare_frame(model, season, totals)
        frames.append(frame)
        result = {
            "test_season": season,
            "training_seasons": list(model.config.training_seasons),
            "training_games": int(model.train_val_test_masks()[0].sum()),
            "forecast": total_forecast_metrics(frame),
            "thresholds": {
                str(threshold): betting_results(frame, threshold)
                for threshold in EDGE_THRESHOLDS
            },
        }
        season_results.append(result)
        baseline = result["thresholds"]["0.0"]
        print(
            f"{season}: total MAE={result['forecast']['total_mae']:.2f}, "
            f"bias={result['forecast']['mean_bias_predicted_minus_actual']:+.2f}, "
            f"O/U={baseline['accuracy']:.1%}, ROI={baseline['roi']:.1%}"
        )

    report = {
        "description": (
            "Expanding-window evaluation of the locked score model against "
            "nflverse closing game totals"
        ),
        "assumed_price": -110,
        "edge_definition": "predicted total minus closing market total",
        "thresholds_predeclared": EDGE_THRESHOLDS,
        "decision_rule": (
            "Choose a threshold only if it is profitable and reasonably "
            "stable in 2022–2023 development, then require it to remain "
            "profitable in 2024–2025 validation."
        ),
        "conclusion": (
            "No totals betting threshold qualifies. Development was "
            "unprofitable at every predeclared threshold. The six-point "
            "threshold was profitable in 2024–2025 but lost in both "
            "development seasons, so it is not a validated product signal."
        ),
        "development": group_summary(frames, [2022, 2023]),
        "validation": group_summary(frames, [2024, 2025]),
        "pooled": group_summary(frames, TEST_SEASONS),
        "seasons": season_results,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
