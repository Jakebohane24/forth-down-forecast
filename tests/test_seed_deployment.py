from datetime import UTC, datetime

from sqlalchemy import func, select

from api.database import Base, create_database
from api.models import Prediction
from scripts.seed_deployment_database import seed_database


def test_seed_database_copies_only_public_model_snapshots(tmp_path):
    source_url = f"sqlite:///{tmp_path / 'source.db'}"
    target_url = f"sqlite:///{tmp_path / 'target.db'}"
    source_engine, source_sessions = create_database(source_url)
    Base.metadata.create_all(source_engine)
    common = {
        "game_id": "2025_06_MIA_BUF",
        "season": 2025,
        "week": 6,
        "home_team": "BUF",
        "away_team": "MIA",
        "predicted_home_score": 27,
        "predicted_away_score": 20,
        "predicted_winner": "BUF",
        "model_win_confidence": 0.66,
        "moneyline_signal": True,
        "model_version": "rolling-through-2024-no-wind-v4-signal-60",
        "generated_at": datetime(2026, 8, 1, tzinfo=UTC),
    }
    with source_sessions.begin() as session:
        session.add(Prediction(snapshot_id="public", **common))
        session.add(
            Prediction(
                snapshot_id="obsolete",
                **{**common, "model_version": "rolling-through-2024-v1"},
            )
        )
    source_engine.dispose()

    first = seed_database(source_url, target_url)
    second = seed_database(source_url, target_url)
    target_engine, target_sessions = create_database(target_url)
    with target_sessions() as session:
        rows = session.scalar(select(func.count()).select_from(Prediction))
        version = session.scalar(select(Prediction.model_version))
    target_engine.dispose()

    assert first["predictions"] == 1
    assert second["predictions"] == 0
    assert rows == 1
    assert version == "rolling-through-2024-no-wind-v4-signal-60"
