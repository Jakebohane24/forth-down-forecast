from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select

from api.database import create_database
from api.ingest import persist_game_results, persist_predictions
from api.models import GameResult, Prediction


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


def test_game_result_is_stored_separately_from_prediction(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'app.db'}"
    games = pd.DataFrame(
        {
            "season": [2025],
            "week": [6],
            "home_team": ["BUF"],
            "away_team": ["MIA"],
            "home_score": [31],
            "away_score": [20],
        },
        index=pd.Index(["2025_06_MIA_BUF"], name="game_id"),
    )

    persist_game_results(games, database_url=database_url)
    _, session_factory = create_database(database_url)
    with session_factory() as session:
        result = session.get(GameResult, "2025_06_MIA_BUF")

    assert result.home_score == 31
    assert result.away_score == 20
