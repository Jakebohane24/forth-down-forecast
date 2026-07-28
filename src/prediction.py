"""Interface-facing weekly prediction orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import EvaluationConfig
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
    ):
        self.model = model
        self.simulation_config = simulation_config or EvaluationConfig()

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
