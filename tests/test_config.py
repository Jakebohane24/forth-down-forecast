import pytest

from src.config import EvaluationConfig, ModelConfig


def test_default_model_configuration_is_explicit():
    config = ModelConfig()

    assert config.use_market_history is True
    assert config.stacking_strategy == "kfold"
    assert config.market_history_features == "all"
    assert config.validation_season == 2024
    assert config.test_season == 2025


def test_model_configuration_round_trip():
    original = ModelConfig(training_seasons=(2020, 2021, 2022))

    restored = ModelConfig.from_dict(original.to_dict())

    assert restored == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stacking_strategy": "random"},
        {"market_history_features": "unknown"},
        {"oof_folds": 1},
        {"tuning_iterations": 0},
    ],
)
def test_invalid_model_configuration_fails_fast(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_invalid_evaluation_configuration_fails_fast():
    with pytest.raises(ValueError):
        EvaluationConfig(simulations=0)
