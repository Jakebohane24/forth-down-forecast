"""Typed configuration for training, simulation, and weekly predictions."""

from dataclasses import asdict, dataclass
from typing import Literal


StackingStrategy = Literal["kfold", "timeseries"]
TuningStrategy = Literal["kfold", "timeseries"]
MarketHistoryFeatures = Literal["corrected", "composite", "all"]


@dataclass(frozen=True)
class ModelConfig:
    """Reproducible settings for the two-stage NFL model."""

    use_market_history: bool = True
    stacking_strategy: StackingStrategy = "kfold"
    tuning_strategy: TuningStrategy = "timeseries"
    market_history_features: MarketHistoryFeatures = "all"
    random_state: int = 24
    oof_folds: int = 5
    tuning_folds: int = 3
    tuning_iterations: int = 20
    training_seasons: tuple[int, ...] = tuple(range(2018, 2024))
    validation_season: int = 2024
    test_season: int = 2025

    def __post_init__(self):
        if self.stacking_strategy not in {"kfold", "timeseries"}:
            raise ValueError("stacking_strategy must be 'kfold' or 'timeseries'")
        if self.tuning_strategy not in {"kfold", "timeseries"}:
            raise ValueError("tuning_strategy must be 'kfold' or 'timeseries'")
        if self.market_history_features not in {
            "corrected",
            "composite",
            "all",
        }:
            raise ValueError(
                "market_history_features must be 'corrected', " "'composite', or 'all'"
            )
        if self.oof_folds < 2 or self.tuning_folds < 2:
            raise ValueError("Cross-validation fold counts must be at least 2")
        if self.tuning_iterations < 1:
            raise ValueError("tuning_iterations must be positive")
        if not self.training_seasons:
            raise ValueError("training_seasons cannot be empty")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["training_seasons"] = tuple(values["training_seasons"])
        return cls(**values)


@dataclass(frozen=True)
class EvaluationConfig:
    simulations: int = 10_000
    random_state: int = 24
    high_edge_threshold: float = 4.0

    def __post_init__(self):
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.high_edge_threshold < 0:
            raise ValueError("high_edge_threshold cannot be negative")


@dataclass(frozen=True)
class BettingConfig:
    """Frozen thresholds used by the public prediction product."""

    moneyline_confidence_threshold: float = 0.625

    def __post_init__(self):
        if not 0.5 <= self.moneyline_confidence_threshold <= 1:
            raise ValueError(
                "moneyline_confidence_threshold must be between 0.5 and 1"
            )
