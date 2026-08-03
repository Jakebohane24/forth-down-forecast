"""Run expanding-window season backtests for the locked model architecture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

from src.config import BettingConfig, EvaluationConfig, ModelConfig
from src.evaluation import evaluate_model, evaluation_frame
from src.training import NFLModel


def american_profit(odds: pd.Series) -> pd.Series:
    """Return profit on a winning one-unit stake at American odds."""
    return pd.Series(
        odds.where(odds > 0, 100).div(100).where(odds > 0, 100 / odds.abs()),
        index=odds.index,
    )


def spread_results(frame: pd.DataFrame, threshold: float) -> dict:
    bets = frame.loc[(~frame["is_push"]) & (frame["edge"] >= threshold)].copy()
    bets["won"] = bets["model_picked_home"] == bets["home_covered"]
    bets["profit"] = bets["won"].map({True: 100 / 110, False: -1.0})
    return {
        "bets": len(bets),
        "wins": int(bets["won"].sum()),
        "losses": int((~bets["won"]).sum()),
        "ats_rate": float(bets["won"].mean()),
        "profit_units": float(bets["profit"].sum()),
        "roi": float(bets["profit"].mean()),
    }


def moneyline_results(
    frame: pd.DataFrame,
    schedules: pd.DataFrame,
    confidence_threshold: float,
    minimum_odds: float,
) -> dict:
    odds_columns = [
        "season",
        "week",
        "home_team",
        "away_team",
        "home_moneyline",
        "away_moneyline",
    ]
    joined = frame.merge(
        schedules[odds_columns],
        on=["season", "week", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )
    joined["model_win_confidence"] = joined["home_win_prob"].where(
        joined["pred_home_win"],
        1 - joined["home_win_prob"],
    )
    available = joined["home_moneyline"].notna() & joined["away_moneyline"].notna()
    joined["odds"] = joined["home_moneyline"].where(
        joined["pred_home_win"],
        joined["away_moneyline"],
    )
    bets = joined.loc[
        available
        & (joined["model_win_confidence"] >= confidence_threshold)
        & (joined["odds"] >= minimum_odds)
    ].copy()
    bets["won"] = (
        bets["pred_home_win"]
        & (bets["home_score"] > bets["away_score"])
    ) | (
        ~bets["pred_home_win"]
        & (bets["away_score"] > bets["home_score"])
    )
    bets["profit"] = american_profit(bets["odds"]).where(bets["won"], -1.0)
    return {
        "bets": len(bets),
        "wins": int(bets["won"].sum()),
        "losses": int((~bets["won"]).sum()),
        "win_rate": float(bets["won"].mean()),
        "profit_units": float(bets["profit"].sum()),
        "roi": float(bets["profit"].mean()),
        "games_missing_odds": int((~available).sum()),
    }


def pooled_results(seasons: list[dict], result_key: str) -> dict:
    """Pool flat-stake betting results across test seasons."""
    results = [season[result_key] for season in seasons]
    bets = sum(result["bets"] for result in results)
    wins = sum(result["wins"] for result in results)
    losses = sum(result["losses"] for result in results)
    profit = sum(result["profit_units"] for result in results)
    return {
        "bets": bets,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / bets,
        "profit_units": profit,
        "roi": profit / bets,
    }


def run_backtest(
    first_test_season: int = 2021,
    last_test_season: int = 2025,
    stacking_strategy: str = "kfold",
    tuning_strategy: str = "timeseries",
    use_wind: bool = True,
) -> dict:
    evaluation_config = EvaluationConfig()
    betting_config = BettingConfig()
    schedules = nfl.import_schedules(
        list(range(first_test_season, last_test_season + 1))
    )
    seasons = []

    for test_season in range(first_test_season, last_test_season + 1):
        training_seasons = tuple(range(2018, test_season))
        config = ModelConfig(
            use_wind=use_wind,
            stacking_strategy=stacking_strategy,
            tuning_strategy=tuning_strategy,
            training_seasons=training_seasons,
            validation_season=test_season - 1,
            test_season=test_season,
        )
        model = NFLModel(config).train()
        frame = evaluation_frame(model, "test", config=evaluation_config)
        metrics = evaluate_model(
            model,
            "test",
            config=evaluation_config,
            verbose=False,
        )
        train_mask, _, _ = model.train_val_test_masks()
        seasons.append(
            {
                "test_season": test_season,
                "training_seasons": list(training_seasons),
                "training_games": int(train_mask.sum()),
                "test_games": len(frame),
                "margin_mae": metrics["margin_mae"],
                "margin_rmse": metrics["margin_rmse"],
                "win_accuracy": metrics["win_accuracy"],
                "ats_accuracy": metrics["ats_accuracy"],
                "spread_4_point": spread_results(
                    frame,
                    evaluation_config.high_edge_threshold,
                ),
                "moneyline_signal": moneyline_results(
                    frame,
                    schedules,
                    betting_config.moneyline_confidence_threshold,
                    betting_config.moneyline_minimum_odds,
                ),
            }
        )
        print(
            f"{test_season}: train N={int(train_mask.sum())}, "
            f"MAE={metrics['margin_mae']:.2f}, "
            f"win={metrics['win_accuracy']:.1%}, "
            f"spread ROI={seasons[-1]['spread_4_point']['roi']:.1%}, "
            f"moneyline ROI="
            f"{seasons[-1]['moneyline_signal']['roi']:.1%}"
        )

    return {
        "description": (
            "Expanding-window backtest of the locked model architecture using "
            f"{stacking_strategy} stage-one out-of-fold stacking and "
            f"{tuning_strategy} hyperparameter cross-validation"
        ),
        "stacking_strategy": stacking_strategy,
        "tuning_strategy": tuning_strategy,
        "use_wind": use_wind,
        "methodology": (
            "Each test season is predicted by a newly tuned model trained only "
            "on processed games from 2018 through the prior season. "
            f"Hyperparameters use {tuning_strategy} cross-validation."
        ),
        "caveat": (
            "Earlier test seasons use materially fewer training games. The "
            "4.0-point spread and the combined 60% confidence / -300 "
            "moneyline rule were selected retrospectively and are not unbiased."
        ),
        "spread_price": -110,
        "moneyline_source": "nflverse schedule closing moneylines",
        "pooled_results": {
            "spread_4_point": pooled_results(seasons, "spread_4_point"),
            "moneyline_signal": pooled_results(
                seasons,
                "moneyline_signal",
            ),
        },
        "seasons": seasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rolling_backtest.json"),
    )
    parser.add_argument("--first-test-season", type=int, default=2021)
    parser.add_argument("--last-test-season", type=int, default=2025)
    parser.add_argument(
        "--stacking-strategy",
        choices=["kfold", "timeseries"],
        default="kfold",
    )
    parser.add_argument(
        "--tuning-strategy",
        choices=["kfold", "timeseries"],
        default="timeseries",
    )
    parser.add_argument(
        "--use-wind",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    results = run_backtest(
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        stacking_strategy=args.stacking_strategy,
        tuning_strategy=args.tuning_strategy,
        use_wind=args.use_wind,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
