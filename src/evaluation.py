"""Simulation and out-of-sample evaluation for trained NFL models."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import EvaluationConfig


DatasetSplit = Literal["val", "test"]
TieHandling = Literal["away", "split"]


def simulate_scores(
    point_predictions: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
    tie_handling: TieHandling = "split",
) -> pd.DataFrame:
    """Convert expected offensive points into game-level score distributions."""
    config = config or EvaluationConfig()

    required = {"pred_home_offense_points", "pred_away_offense_points"}
    missing = sorted(required.difference(point_predictions.columns))
    if missing:
        raise ValueError(f"Point predictions are missing columns: {missing}")

    rng = np.random.default_rng(config.random_state)
    game_count = len(point_predictions)
    simulations = config.simulations

    home_points = point_predictions["pred_home_offense_points"].clip(lower=0).to_numpy()
    away_points = point_predictions["pred_away_offense_points"].clip(lower=0).to_numpy()
    home_td_lambda = ((home_points * 0.78) / 7.0)[:, None]
    away_td_lambda = ((away_points * 0.78) / 7.0)[:, None]
    home_fg_lambda = ((home_points * 0.22) / 3.0)[:, None]
    away_fg_lambda = ((away_points * 0.22) / 3.0)[:, None]

    # Preserve the established seeded draw order so historical evaluations are
    # exactly reproducible after moving simulation out of the training module.
    home_offense_tds = rng.poisson(home_td_lambda, size=(game_count, simulations))
    away_offense_tds = rng.poisson(away_td_lambda, size=(game_count, simulations))
    home_non_offense_tds = rng.poisson(0.1428, size=(game_count, simulations))
    away_non_offense_tds = rng.poisson(0.1428, size=(game_count, simulations))
    home_field_goals = rng.poisson(home_fg_lambda, size=(game_count, simulations))
    away_field_goals = rng.poisson(away_fg_lambda, size=(game_count, simulations))
    home_scores = (home_offense_tds + home_non_offense_tds) * 7 + home_field_goals * 3
    away_scores = (away_offense_tds + away_non_offense_tds) * 7 + away_field_goals * 3

    margin = home_scores - away_scores
    total = home_scores + away_scores
    if tie_handling not in {"away", "split"}:
        raise ValueError("tie_handling must be 'away' or 'split'")
    home_win_probability = np.mean(margin > 0, axis=1)
    tie_probability = np.mean(margin == 0, axis=1)
    if tie_handling == "split":
        home_win_probability = home_win_probability + 0.5 * tie_probability

    return pd.DataFrame(
        {
            "pred_home_score": np.median(home_scores, axis=1).astype(int),
            "pred_away_score": np.median(away_scores, axis=1).astype(int),
            "pred_spread": np.round(np.median(margin, axis=1) * 2) / 2,
            "pred_over_under": np.round(np.mean(total, axis=1) * 2) / 2,
            "home_win_prob": home_win_probability,
            "tie_prob": tie_probability,
            "pred_home_win": home_win_probability > 0.5,
        },
        index=point_predictions.index,
    )


def evaluation_frame(
    model,
    split: DatasetSplit,
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Return game metadata, predictions, outcomes, and betting comparisons."""
    train_mask, val_mask, test_mask = model.train_val_test_masks()
    if split == "val":
        mask = val_mask
    elif split == "test":
        mask = test_mask
    else:
        raise ValueError("split must be 'val' or 'test'")

    predictions = simulate_scores(
        model.predict(split),
        config=config,
    )
    columns = [
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
    ]
    frame = pd.concat([model.df.loc[mask, columns], predictions], axis=1)
    frame["actual_margin"] = frame["home_score"] - frame["away_score"]
    frame["actual_home_win"] = frame["actual_margin"] > 0
    frame["home_covered"] = frame["actual_margin"] > frame["spread_line"]
    frame["model_picked_home"] = frame["pred_spread"] > frame["spread_line"]
    frame["edge"] = (frame["pred_spread"] - frame["spread_line"]).abs()
    frame["is_push"] = frame["actual_margin"] == frame["spread_line"]
    return frame


def evaluate_model(
    model,
    split: DatasetSplit,
    *,
    config: EvaluationConfig | None = None,
    verbose: bool = True,
) -> dict:
    """Evaluate one trained model on an untouched validation or test split."""
    config = config or EvaluationConfig()
    frame = evaluation_frame(model, split, config=config)
    actual_margin = frame["actual_margin"]
    no_pushes = frame.loc[~frame["is_push"]]
    high_edge = no_pushes.loc[no_pushes["edge"] >= config.high_edge_threshold]

    ats_accuracy = (no_pushes["model_picked_home"] == no_pushes["home_covered"]).mean()
    high_edge_accuracy = (
        (high_edge["model_picked_home"] == high_edge["home_covered"]).mean()
        if not high_edge.empty
        else float("nan")
    )

    metrics = {
        "split": split,
        "games": len(frame),
        "home_score_mae": mean_absolute_error(
            frame["home_score"], frame["pred_home_score"]
        ),
        "away_score_mae": mean_absolute_error(
            frame["away_score"], frame["pred_away_score"]
        ),
        "margin_mae": mean_absolute_error(actual_margin, frame["pred_spread"]),
        "margin_rmse": mean_squared_error(actual_margin, frame["pred_spread"]) ** 0.5,
        "win_accuracy": (frame["pred_home_win"] == frame["actual_home_win"]).mean(),
        "ats_accuracy": ats_accuracy,
        "high_edge_threshold": config.high_edge_threshold,
        "high_edge_games": len(high_edge),
        "high_edge_ats_accuracy": high_edge_accuracy,
    }

    if verbose:
        print(
            f"{split}: margin MAE={metrics['margin_mae']:.3f}, "
            f"win={metrics['win_accuracy']:.2%}, "
            f"ATS={metrics['ats_accuracy']:.2%}, "
            f"high-edge ATS={metrics['high_edge_ats_accuracy']:.2%} "
            f"(N={metrics['high_edge_games']})"
        )
    return metrics
