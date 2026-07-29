from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import create_app
from api.models import Prediction


def test_health_model_and_performance_endpoints(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}")

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/model").status_code == 200
        performance = client.get("/performance").json()

    assert performance["moneyline_threshold"] == 0.625
    assert performance["pooled_bets"] == 280


def test_week_endpoint_returns_latest_snapshot(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}")

    with TestClient(app) as client:
        with app.state.session_factory() as session:
            common = {
                "game_id": "2026_01_MIA_BUF",
                "season": 2026,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "predicted_home_score": 27,
                "predicted_away_score": 20,
                "predicted_winner": "BUF",
                "model_win_confidence": 0.68,
                "moneyline_signal": True,
                "model_version": "production-2018-2025-v1",
                "generated_at": datetime(2026, 9, 1, tzinfo=UTC),
            }
            session.add(Prediction(snapshot_id="old", **common))
            session.add(
                Prediction(
                    snapshot_id="new",
                    **{
                        **common,
                        "generated_at": datetime(2026, 9, 2, tzinfo=UTC),
                    },
                )
            )
            session.commit()

        result = client.get("/predictions/2026/1").json()

    assert result["count"] == 1
    assert result["predictions"][0]["predicted_winner"] == "BUF"
