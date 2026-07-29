"""Compare moneyline trigger rules on saved expanding-window predictions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


CONFIDENCE_THRESHOLDS = [
    0.50,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
]


def american_implied_probability(odds: pd.Series) -> pd.Series:
    """Convert American odds to their unadjusted implied probability."""
    return pd.Series(
        np.where(
            odds < 0,
            odds.abs() / (odds.abs() + 100),
            100 / (odds + 100),
        ),
        index=odds.index,
        dtype=float,
    )


def american_profit(odds: pd.Series) -> pd.Series:
    """Return profit on a winning one-unit stake at American odds."""
    return pd.Series(
        np.where(odds > 0, odds / 100, 100 / odds.abs()),
        index=odds.index,
        dtype=float,
    )


def load_games(database: Path) -> pd.DataFrame:
    """Load one saved expanding-window prediction and result per game."""
    query = """
        SELECT
            p.game_id,
            p.season,
            p.week,
            p.predicted_winner,
            p.model_win_confidence,
            p.home_team,
            p.away_team,
            p.home_moneyline,
            p.away_moneyline,
            r.home_score,
            r.away_score
        FROM predictions AS p
        JOIN game_results AS r USING (game_id)
        WHERE p.season BETWEEN 2022 AND 2025
          AND p.home_moneyline IS NOT NULL
          AND p.away_moneyline IS NOT NULL
    """
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(query, connection)

    if frame["game_id"].duplicated().any():
        raise ValueError("Expected one saved prediction per game")

    frame["picked_home"] = frame["predicted_winner"] == frame["home_team"]
    frame["picked_odds"] = frame["home_moneyline"].where(
        frame["picked_home"],
        frame["away_moneyline"],
    )
    frame["won"] = (
        frame["picked_home"] & (frame["home_score"] > frame["away_score"])
    ) | (
        ~frame["picked_home"] & (frame["away_score"] > frame["home_score"])
    )
    home_implied = american_implied_probability(frame["home_moneyline"])
    away_implied = american_implied_probability(frame["away_moneyline"])
    market_hold = home_implied + away_implied
    frame["picked_market_probability"] = home_implied.where(
        frame["picked_home"],
        away_implied,
    ) / market_hold
    frame["model_market_edge"] = (
        frame["model_win_confidence"] - frame["picked_market_probability"]
    )
    frame["picked_market_favorite"] = frame["picked_market_probability"] >= 0.5
    frame["profit"] = american_profit(frame["picked_odds"]).where(
        frame["won"],
        -1.0,
    )
    frame["market_favorite_home"] = home_implied >= away_implied
    frame["market_favorite_odds"] = frame["home_moneyline"].where(
        frame["market_favorite_home"],
        frame["away_moneyline"],
    )
    frame["market_favorite_won"] = (
        frame["market_favorite_home"]
        & (frame["home_score"] > frame["away_score"])
    ) | (
        ~frame["market_favorite_home"]
        & (frame["away_score"] > frame["home_score"])
    )
    frame["market_favorite_profit"] = american_profit(
        frame["market_favorite_odds"]
    ).where(frame["market_favorite_won"], -1.0)
    return frame


def summarize(frame: pd.DataFrame) -> dict:
    """Summarize flat one-unit betting results."""
    bets = len(frame)
    profit = float(frame["profit"].sum())
    return {
        "bets": bets,
        "wins": int(frame["won"].sum()),
        "win_rate": float(frame["won"].mean()) if bets else None,
        "profit_units": profit,
        "roi": profit / bets if bets else None,
    }


def evaluate_rule(frame: pd.DataFrame, mask: pd.Series) -> dict:
    """Evaluate one rule in development, validation, pooled, and yearly views."""
    bets = frame.loc[mask].copy()
    favorite_control = bets.copy()
    favorite_control["won"] = favorite_control["market_favorite_won"]
    favorite_control["profit"] = favorite_control["market_favorite_profit"]
    yearly = {
        str(season): summarize(bets.loc[bets["season"] == season])
        for season in range(2022, 2026)
    }
    yearly_rois = [
        result["roi"] for result in yearly.values() if result["roi"] is not None
    ]
    return {
        "pooled_2022_2025": summarize(bets),
        "development_2022_2023": summarize(
            bets.loc[bets["season"].isin([2022, 2023])]
        ),
        "validation_2024_2025": summarize(
            bets.loc[bets["season"].isin([2024, 2025])]
        ),
        "same_games_market_favorite_control": summarize(favorite_control),
        "model_market_disagreements": int((~bets["picked_market_favorite"]).sum()),
        "profitable_seasons": sum(roi > 0 for roi in yearly_rois),
        "worst_season_roi": min(yearly_rois) if yearly_rois else None,
        "by_season": yearly,
    }


def build_report(frame: pd.DataFrame) -> dict:
    """Build a transparent comparison without selecting on pooled ROI alone."""
    confidence = {
        f"{threshold:.3f}": evaluate_rule(
            frame,
            frame["model_win_confidence"] >= threshold,
        )
        for threshold in CONFIDENCE_THRESHOLDS
    }
    alternatives = {
        "all_model_picks": evaluate_rule(frame, pd.Series(True, index=frame.index)),
        "market_favorite_all": evaluate_rule(
            frame,
            frame["picked_market_favorite"],
        ),
        "confidence_625_favorites": evaluate_rule(
            frame,
            (frame["model_win_confidence"] >= 0.625)
            & frame["picked_market_favorite"],
        ),
        "confidence_625_underdogs": evaluate_rule(
            frame,
            (frame["model_win_confidence"] >= 0.625)
            & ~frame["picked_market_favorite"],
        ),
        "confidence_650_favorites": evaluate_rule(
            frame,
            (frame["model_win_confidence"] >= 0.65)
            & frame["picked_market_favorite"],
        ),
        "confidence_650_underdogs": evaluate_rule(
            frame,
            (frame["model_win_confidence"] >= 0.65)
            & ~frame["picked_market_favorite"],
        ),
        "positive_model_market_edge": evaluate_rule(
            frame,
            frame["model_market_edge"] >= 0,
        ),
        "model_market_edge_5_points": evaluate_rule(
            frame,
            frame["model_market_edge"] >= 0.05,
        ),
        "confidence_625_and_positive_edge": evaluate_rule(
            frame,
            (frame["model_win_confidence"] >= 0.625)
            & (frame["model_market_edge"] >= 0),
        ),
    }
    return {
        "description": (
            "Moneyline trigger comparison on expanding-window test predictions "
            "for 2022–2025"
        ),
        "sample": {
            "games": len(frame),
            "weeks": "6–18",
            "odds": "nflverse closing moneylines",
            "staking": "flat one-unit stakes",
        },
        "selection_policy": (
            "Use 2022–2023 as development and 2024–2025 as validation; prefer "
            "adequate bet count and yearly stability over the highest pooled ROI."
        ),
        "confidence_thresholds": confidence,
        "alternative_rules": alternatives,
        "warnings": [
            "Closing odds differ from the planned 24-hours-before-kickoff odds.",
            "Thresholds inspected here are retrospective and require prospective validation.",
            "Model confidence is not guaranteed to be a calibrated probability.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/app.db"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/moneyline_thresholds_2022_2025.json"),
    )
    args = parser.parse_args()
    report = build_report(load_games(args.database))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
