import pandas as pd
import pytest

from src.config import EvaluationConfig
from src.market import StaticMarketDataProvider
from src.prediction import WeeklyPredictor


class FakeModel:
    def predict_features(self, features):
        return pd.DataFrame(
            {
                "pred_home_offense_points": [24.0],
                "pred_away_offense_points": [20.0],
            },
            index=features.index,
        )


def test_weekly_prediction_adds_consensus_market_edge():
    features = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
        },
        index=pd.Index(["2026_01_MIA_BUF"], name="game_id"),
    )
    lines = pd.DataFrame(
        {
            "game_id": ["2026_01_MIA_BUF"],
            "sportsbook": ["Book A"],
            "market_home_margin": [2.5],
            "home_spread_price": [-110],
            "away_spread_price": [-110],
            "total": [47.5],
            "home_moneyline": [-145],
            "away_moneyline": [125],
            "retrieved_at": ["2026-09-01T12:00:00Z"],
        }
    )
    predictor = WeeklyPredictor(
        FakeModel(),
        simulation_config=EvaluationConfig(
            simulations=100,
            random_state=7,
        ),
    )

    result = predictor.predict(
        features,
        market_provider=StaticMarketDataProvider(lines),
    )

    assert result.loc["2026_01_MIA_BUF", "sportsbook"] == "consensus"
    assert result.loc["2026_01_MIA_BUF", "model_edge"] == (
        result.loc["2026_01_MIA_BUF", "pred_spread"] - 2.5
    )


def test_prediction_snapshot_refuses_to_overwrite(tmp_path):
    path = tmp_path / "snapshot.parquet"
    frame = pd.DataFrame({"value": [1]}, index=pd.Index(["game_a"]))

    WeeklyPredictor.save_snapshot(frame, path)

    with pytest.raises(FileExistsError):
        WeeklyPredictor.save_snapshot(frame, path)
