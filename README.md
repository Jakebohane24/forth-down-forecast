# NFL Project 2

Two-stage NFL score prediction project with reproducible processing,
training, evaluation, model artifacts, and weekly prediction contracts.

## Current model

```python
from src.config import ModelConfig
from src.training import NFLModel

config = ModelConfig(
    use_market_history=True,
    stacking_strategy="kfold",
    market_history_features="all",
)
model = NFLModel(config).train()
```

Stage one predicts team efficiency and volume components. Stage two uses
out-of-fold stage-one predictions and rolling game features to predict home
and away offensive points.

## Evaluate

```python
from src.evaluation import evaluate_model

evaluate_model(model, "val")
evaluate_model(model, "test")
```

The locked regression baseline is stored in
`reports/baseline_metrics.json`.

The current high-edge threshold is a model-versus-market difference of at
least 4.0 points. It is an experimental indicator, not a proven betting
recommendation.

Retrospective spread and moneyline indicator results are recorded in
`reports/betting_retrospective.json`. The moneyline experiment uses a 62.5%
model-confidence threshold. Both thresholds were chosen after inspecting the
2024 and 2025 results, so they must be tracked prospectively before being
treated as reliable.

## Save and load

```python
model.save("artifacts/models/default")
loaded = NFLModel.load("artifacts/models/default")
```

Generated model artifacts and the local SQLite database are intentionally
excluded from Git.

## Weekly predictions

`src.prediction.WeeklyPredictor` accepts prepared pregame matchup features,
runs the saved model, simulates score distributions, joins normalized market
lines, and saves immutable prediction snapshots.

`src.market.MarketDataProvider` isolates the application from any particular
odds vendor. Current internal spread convention is a positive
`market_home_margin` when the home team is favored.

## Tests

```bash
pytest -q
```

The next pipeline component is a pregame feature builder that converts an
upcoming schedule and each team's latest completed-game history into the same
feature schema used by the trained model.
