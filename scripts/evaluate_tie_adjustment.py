"""Compare legacy and split-tie probabilities on rolling no-wind models."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BettingConfig, EvaluationConfig
from src.evaluation import simulate_scores
from src.training import NFLModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE = PROJECT_ROOT / "data" / "app.db"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models" / "no_wind_rolling"
OUTPUT = PROJECT_ROOT / "reports" / "tie_adjustment_2022_2025.json"
OFFICIAL_REPORT = PROJECT_ROOT / "reports" / "rolling_backtest_no_wind_2022_2025.json"


def american_profit(odds: pd.Series) -> pd.Series:
    return odds.where(odds > 0, 100).div(100).where(odds > 0, 100 / odds.abs())


def assess(
    frame: pd.DataFrame,
    probability: pd.Series,
    *,
    confidence_threshold: float | None = None,
    minimum_odds: float | None = None,
) -> tuple[dict, pd.Series, pd.Series]:
    config = BettingConfig()
    confidence_threshold = (
        config.moneyline_confidence_threshold
        if confidence_threshold is None
        else confidence_threshold
    )
    minimum_odds = (
        config.moneyline_minimum_odds if minimum_odds is None else minimum_odds
    )
    picked_home = probability > 0.5
    confidence = probability.where(picked_home, 1 - probability)
    odds = frame["home_moneyline"].where(picked_home, frame["away_moneyline"])
    signal = (
        (confidence >= confidence_threshold)
        & odds.notna()
        & (odds >= minimum_odds)
    )
    correct = picked_home == frame["actual_home_win"]
    returns = american_profit(odds.loc[signal]).where(correct.loc[signal], -1.0)
    metrics = {
        "games": len(frame),
        "win_accuracy": float(correct.mean()),
        "brier_score": float(np.mean((probability - frame["actual_home_win"]) ** 2)),
        "mean_confidence": float(confidence.mean()),
        "signals": int(signal.sum()),
        "signal_wins": int(correct.loc[signal].sum()),
        "signal_accuracy": float(correct.loc[signal].mean()) if signal.any() else None,
        "signal_profit_units": float(returns.sum()),
        "signal_roi": float(returns.mean()) if signal.any() else None,
    }
    return metrics, picked_home, signal


def main() -> None:
    evaluation_config = EvaluationConfig()
    seasons = []
    season_inputs = []
    with sqlite3.connect(DATABASE) as connection:
        for season in range(2022, 2026):
            model = NFLModel.load(ARTIFACT_ROOT / str(season))
            _, _, test_mask = model.train_val_test_masks()
            metadata = model.df.loc[
                test_mask,
                ["season", "week", "home_team", "away_team", "home_score", "away_score"],
            ].copy()
            odds = pd.read_sql_query(
                """
                SELECT game_id, home_moneyline, away_moneyline
                FROM predictions
                WHERE season = ? AND model_version LIKE '%-no-wind-v4-signal-60'
                """,
                connection,
                params=(season,),
                index_col="game_id",
            )
            frame = metadata.join(odds, how="left")
            frame["actual_home_win"] = frame["home_score"] > frame["away_score"]

            points = model.predict("test")
            legacy = simulate_scores(
                points,
                config=evaluation_config,
                tie_handling="away",
            )
            adjusted = simulate_scores(
                points,
                config=evaluation_config,
                tie_handling="split",
            )
            legacy_metrics, legacy_pick, legacy_signal = assess(
                frame, legacy["home_win_prob"]
            )
            adjusted_metrics, adjusted_pick, adjusted_signal = assess(
                frame, adjusted["home_win_prob"]
            )
            seasons.append(
                {
                    "season": season,
                    "mean_simulated_tie_probability": float(legacy["tie_prob"].mean()),
                    "legacy": legacy_metrics,
                    "split_ties": adjusted_metrics,
                    "picks_changed": int((legacy_pick != adjusted_pick).sum()),
                    "signals_added": int((~legacy_signal & adjusted_signal).sum()),
                    "signals_removed": int((legacy_signal & ~adjusted_signal).sum()),
                }
            )
            season_inputs.append((season, frame, adjusted["home_win_prob"]))

    pooled = {}
    for method in ("legacy", "split_ties"):
        bets = sum(row[method]["signals"] for row in seasons)
        wins = sum(row[method]["signal_wins"] for row in seasons)
        profit = sum(row[method]["signal_profit_units"] for row in seasons)
        games = sum(row[method]["games"] for row in seasons)
        pooled[method] = {
            "games": games,
            "signals": bets,
            "signal_wins": wins,
            "signal_accuracy": wins / bets,
            "signal_profit_units": profit,
            "signal_roi": profit / bets,
            "weighted_brier_score": sum(
                row[method]["brier_score"] * row[method]["games"] for row in seasons
            ) / games,
        }

    threshold_grid = []
    for confidence_threshold in (0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70):
        for minimum_odds in (-500, -400, -300, -250, -200, -150, -100, 100):
            yearly = []
            for season, frame, probability in season_inputs:
                metrics, _, _ = assess(
                    frame,
                    probability,
                    confidence_threshold=confidence_threshold,
                    minimum_odds=minimum_odds,
                )
                yearly.append(
                    {
                        "season": season,
                        "signals": metrics["signals"],
                        "wins": metrics["signal_wins"],
                        "accuracy": metrics["signal_accuracy"],
                        "profit_units": metrics["signal_profit_units"],
                        "roi": metrics["signal_roi"],
                    }
                )
            bets = sum(row["signals"] for row in yearly)
            wins = sum(row["wins"] for row in yearly)
            profit = sum(row["profit_units"] for row in yearly)
            threshold_grid.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "minimum_odds": minimum_odds,
                    "signals": bets,
                    "wins": wins,
                    "accuracy": wins / bets if bets else None,
                    "profit_units": profit,
                    "roi": profit / bets if bets else None,
                    "profitable_seasons": sum(row["profit_units"] > 0 for row in yearly),
                    "minimum_season_signals": min(row["signals"] for row in yearly),
                    "yearly": yearly,
                }
            )

    report = {
        "description": (
            "Split simulated ties equally between home and away win probability; "
            "all models, score draws, thresholds, and closing odds are unchanged."
        ),
        "seasons": seasons,
        "pooled": pooled,
        "threshold_grid": threshold_grid,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    selected = next(
        row
        for row in threshold_grid
        if row["confidence_threshold"] == 0.60 and row["minimum_odds"] == -300
    )
    official = json.loads(OFFICIAL_REPORT.read_text(encoding="utf-8"))
    official["description"] += (
        "; simulated score ties are split equally when estimating win probability"
    )
    official["caveat"] = (
        "Earlier test seasons use materially fewer training games. The 4.0-point "
        "spread and combined 60% confidence / -300 moneyline rule were selected "
        "retrospectively and are not unbiased."
    )
    for season_row, adjusted_row, selected_row in zip(
        official["seasons"], seasons, selected["yearly"], strict=True
    ):
        season_row["win_accuracy"] = adjusted_row["split_ties"]["win_accuracy"]
        season_row["moneyline_signal"] = {
            "bets": selected_row["signals"],
            "wins": selected_row["wins"],
            "losses": selected_row["signals"] - selected_row["wins"],
            "win_rate": selected_row["accuracy"],
            "profit_units": selected_row["profit_units"],
            "roi": selected_row["roi"],
            "games_missing_odds": 0,
        }
    official["pooled_results"]["moneyline_signal"] = {
        "bets": selected["signals"],
        "wins": selected["wins"],
        "losses": selected["signals"] - selected["wins"],
        "win_rate": selected["accuracy"],
        "profit_units": selected["profit_units"],
        "roi": selected["roi"],
    }
    OFFICIAL_REPORT.write_text(json.dumps(official, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
