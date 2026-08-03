from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import create_app
from api.models import GameCondition, GameResult, Prediction


def test_health_model_and_performance_endpoints(tmp_path):
    app = create_app(f"sqlite:///{tmp_path / 'api.db'}")

    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/model").status_code == 200
        performance = client.get("/performance").json()

    assert performance["moneyline_threshold"] == 0.625
    assert performance["moneyline_minimum_odds"] == -300
    assert performance["pooled_bets"] == 93
    assert performance["seasons"][-1]["season"] == 2026
    assert performance["seasons"][-1]["training_games"] == 1476
    assert performance["seasons"][-1]["evaluated_games"] == 0


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
                "moneyline_signal_odds": -200,
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
            session.add(
                GameResult(
                    game_id="2026_01_MIA_BUF",
                    season=2026,
                    week=1,
                    home_team="BUF",
                    away_team="MIA",
                    home_score=30,
                    away_score=17,
                    completed=True,
                    updated_at=datetime(2026, 9, 3, tzinfo=UTC),
                )
            )
            session.add(
                GameCondition(
                    game_id="2026_01_MIA_BUF",
                    venue_name="Highmark Stadium",
                    venue_type="outdoor",
                    roof_status="open",
                    forecast_for=datetime(2026, 9, 3, 17, tzinfo=UTC),
                    retrieved_at=datetime(2026, 9, 2, 17, tzinfo=UTC),
                    wind_mph=11,
                    wind_gust_mph=17,
                    temperature_f=62,
                    precipitation_probability=20,
                    source="open-meteo",
                )
            )
            session.commit()

        result = client.get("/predictions/2026/1").json()
        live_performance = client.get("/performance").json()["seasons"][-1]

    assert result["count"] == 1
    assert result["predictions"][0]["predicted_winner"] == "BUF"
    assert result["predictions"][0]["actual_home_score"] == 30
    assert result["predictions"][0]["prediction_correct"] is True
    assert result["predictions"][0]["wind_mph"] == 11
    assert result["predictions"][0]["weather_source"] == "open-meteo"
    assert live_performance["prediction_count"] == 1
    assert live_performance["evaluated_games"] == 1
    assert live_performance["moneyline_bets"] == 1
    assert live_performance["moneyline_accuracy"] == 1.0
    assert live_performance["margin_mae"] == 6.0
    assert live_performance["win_accuracy"] == 1.0
    assert live_performance["moneyline_roi"] == 0.5
