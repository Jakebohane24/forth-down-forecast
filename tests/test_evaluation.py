import pandas as pd
from pandas.testing import assert_frame_equal

from src.config import EvaluationConfig
from src.evaluation import simulate_scores


def test_score_simulation_is_reproducible():
    points = pd.DataFrame(
        {
            "pred_home_offense_points": [21.5, 27.0],
            "pred_away_offense_points": [20.0, 17.5],
        },
        index=["game_a", "game_b"],
    )
    config = EvaluationConfig(simulations=100, random_state=7)

    first = simulate_scores(points, config=config)
    second = simulate_scores(points, config=config)

    assert_frame_equal(first, second)


def test_negative_point_predictions_are_clipped_before_simulation():
    points = pd.DataFrame(
        {
            "pred_home_offense_points": [-3.0],
            "pred_away_offense_points": [-1.0],
        },
        index=["game_a"],
    )

    result = simulate_scores(
        points,
        config=EvaluationConfig(simulations=10, random_state=7),
    )

    assert result.loc["game_a", "pred_home_score"] >= 0
    assert result.loc["game_a", "pred_away_score"] >= 0
