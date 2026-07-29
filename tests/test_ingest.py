from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select

from api.database import create_database
from api.ingest import persist_predictions
from api.models import Prediction


def test_prediction_snapshot_is_persisted(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    frame = pd.DataFrame(
        {
            "season": [2026],
            "week": [6],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "pred_home_score": [27],
            "pred_away_score": [20],
            "predicted_winner": ["BUF"],
            "model_win_confidence": [0.66],
            "moneyline_signal": [True],
            "prediction_timestamp": [datetime(2026, 10, 1, tzinfo=UTC)],
        },
        index=pd.Index(["2026_06_MIA_BUF"], name="game_id"),
    )

    inserted = persist_predictions(
        frame,
        snapshot_id="snapshot-1",
        model_version="production-v1",
        database_url=database_url,
    )
    _, session_factory = create_database(database_url)
    with session_factory() as session:
        count = session.scalar(select(func.count()).select_from(Prediction))

    assert inserted == 1
    assert count == 1
