"""Interface-facing weekly prediction orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import BettingConfig, EvaluationConfig
from src.evaluation import simulate_scores
from src.market import MarketDataProvider, consensus_lines


PREDICTION_METADATA = ["season", "week", "home_team", "away_team"]


class WeeklyPredictor:
    """Generate and persist predictions from prepared pregame feature rows."""

    def __init__(
        self,
        model,
        *,
        simulation_config: EvaluationConfig | None = None,
        betting_config: BettingConfig | None = None,
    ):
        self.model = model
        self.simulation_config = simulation_config or EvaluationConfig()
        self.betting_config = betting_config or BettingConfig()

    def predict(
        self,
        matchup_features: pd.DataFrame,
        *,
        market_provider: MarketDataProvider | None = None,
    ) -> pd.DataFrame:
        missing = sorted(set(PREDICTION_METADATA).difference(matchup_features.columns))
        if missing:
            raise ValueError(f"Matchup rows are missing metadata: {missing}")
        if not matchup_features.index.is_unique:
            raise ValueError("Matchup feature game IDs must be unique.")

        point_predictions = self.model.predict_features(matchup_features)
        simulations = simulate_scores(
            point_predictions,
            config=self.simulation_config,
        )
        result = pd.concat(
            [matchup_features[PREDICTION_METADATA], simulations],
            axis=1,
        )
        result.index.name = "game_id"
        result["prediction_timestamp"] = datetime.now(UTC)
        result["predicted_winner"] = result["home_team"].where(
            result["pred_home_win"],
            result["away_team"],
        )
        result["model_win_confidence"] = result["home_win_prob"].where(
            result["pred_home_win"],
            1 - result["home_win_prob"],
        )
        result["moneyline_signal"] = False
        result["moneyline_signal_team"] = None
        result["moneyline_signal_odds"] = None

        if market_provider is not None:
            lines = consensus_lines(
                market_provider.get_lines(matchup_features)
            ).set_index("game_id")
            result = result.join(lines, how="left")
            result["model_edge"] = result["pred_spread"] - result["market_home_margin"]
            result["model_ats_side"] = result["model_edge"].map(
                lambda edge: (
                    "home"
                    if pd.notna(edge) and edge > 0
                    else "away" if pd.notna(edge) and edge < 0 else "none"
                )
            )
            selected_moneyline = result["home_moneyline"].where(
                result["pred_home_win"],
                result["away_moneyline"],
            )
            result["moneyline_signal"] = (
                (
                    result["model_win_confidence"]
                    >= self.betting_config.moneyline_confidence_threshold
                )
                & selected_moneyline.notna()
                & (
                    selected_moneyline
                    >= self.betting_config.moneyline_minimum_odds
                )
            )
            result["moneyline_signal_team"] = result["predicted_winner"].where(
                result["moneyline_signal"]
            )
            result["moneyline_signal_odds"] = selected_moneyline.where(
                result["moneyline_signal"]
            )

        return result

    @staticmethod
    def save_snapshot(predictions: pd.DataFrame, path):
        """Save an immutable, timestamped weekly prediction snapshot."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite prediction snapshot: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.reset_index().to_parquet(path, index=False)
        return path
