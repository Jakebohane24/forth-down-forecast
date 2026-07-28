import pandas as pd
import pytest

from src.market import StaticMarketDataProvider, consensus_lines


def sample_lines():
    return pd.DataFrame(
        {
            "game_id": ["game_a", "game_a"],
            "sportsbook": ["Book A", "Book B"],
            "market_home_margin": [3.0, 2.5],
            "home_spread_price": [-110, -108],
            "away_spread_price": [-110, -112],
            "total": [47.5, 48.0],
            "home_moneyline": [-150, -145],
            "away_moneyline": [130, 125],
            "retrieved_at": [
                "2026-09-01T12:00:00Z",
                "2026-09-01T12:01:00Z",
            ],
        }
    )


def test_consensus_uses_median_line_and_latest_timestamp():
    result = consensus_lines(sample_lines()).set_index("game_id")

    assert result.loc["game_a", "market_home_margin"] == 2.75
    assert result.loc["game_a", "sportsbook"] == "consensus"
    assert str(result.loc["game_a", "retrieved_at"]) == ("2026-09-01 12:01:00+00:00")


def test_static_provider_filters_to_requested_games():
    provider = StaticMarketDataProvider(sample_lines())
    games = pd.DataFrame(index=["game_a"])

    result = provider.get_lines(games)

    assert set(result["game_id"]) == {"game_a"}


def test_duplicate_book_rows_are_rejected():
    duplicated = pd.concat([sample_lines().iloc[[0]]] * 2)

    with pytest.raises(ValueError):
        StaticMarketDataProvider(duplicated)
