import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from src.pregame import build_pregame_features
from src.processing import DEFAULT_DB_PATH, get_features
from src.training import NFLModel


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="Local feature DB unavailable")
def test_pregame_builder_recreates_historical_model_features():
    model = NFLModel()
    historical = get_features()
    target = historical.loc["2025_13_DEN_WAS"]
    schedule = pd.DataFrame(
        {
            "game_id": ["2025_13_DEN_WAS"],
            "season": [2025],
            "week": [13],
            "home_team": ["WAS"],
            "away_team": ["DEN"],
            "div_game": [target["div_game"]],
            "game_wind": [target["game_wind"]],
        }
    )

    result, unavailable = build_pregame_features(
        schedule,
        required_features=model.required_prediction_features(),
    )

    assert unavailable.empty
    expected = target[model.required_prediction_features()].astype(float)
    actual = result.loc["2025_13_DEN_WAS", expected.index].astype(float)
    assert_series_equal(actual, expected, check_names=False, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not DEFAULT_DB_PATH.exists(), reason="Local feature DB unavailable")
def test_pregame_builder_explains_early_season_unavailability():
    schedule = pd.DataFrame(
        {
            "game_id": ["2026_01_MIA_BUF"],
            "season": [2026],
            "week": [1],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
        }
    )

    features, unavailable = build_pregame_features(schedule)

    assert features.empty
    assert "Insufficient current-season history" in unavailable.iloc[0]["reason"]
