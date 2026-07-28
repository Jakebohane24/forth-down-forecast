"""Provider-independent market-line contracts for weekly predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


MARKET_COLUMNS = [
    "game_id",
    "sportsbook",
    "market_home_margin",
    "home_spread_price",
    "away_spread_price",
    "total",
    "home_moneyline",
    "away_moneyline",
    "retrieved_at",
]


def validate_market_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize lines returned by any external provider."""
    missing = sorted(set(MARKET_COLUMNS).difference(lines.columns))
    if missing:
        raise ValueError(f"Market lines are missing columns: {missing}")

    normalized = lines.loc[:, MARKET_COLUMNS].copy()
    normalized["retrieved_at"] = pd.to_datetime(
        normalized["retrieved_at"], utc=True, errors="raise"
    )
    if normalized["game_id"].isna().any():
        raise ValueError("Market lines contain missing game IDs.")
    if normalized.duplicated(["game_id", "sportsbook"]).any():
        raise ValueError("Market lines contain duplicate game/sportsbook rows.")
    return normalized


class MarketDataProvider(ABC):
    """Interface implemented by live, cached, or manually supplied providers."""

    @abstractmethod
    def get_lines(self, games: pd.DataFrame) -> pd.DataFrame:
        """Return normalized lines for the supplied upcoming games."""


class StaticMarketDataProvider(MarketDataProvider):
    """Provider for tests, saved snapshots, and manual CSV imports."""

    def __init__(self, lines: pd.DataFrame):
        self.lines = validate_market_lines(lines)

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(Path(path)))

    def get_lines(self, games: pd.DataFrame) -> pd.DataFrame:
        game_ids = set(games.index.astype(str))
        return self.lines.loc[self.lines["game_id"].isin(game_ids)].copy()


def consensus_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Create one median market line per game from available sportsbooks."""
    lines = validate_market_lines(lines)
    numeric = [
        "market_home_margin",
        "home_spread_price",
        "away_spread_price",
        "total",
        "home_moneyline",
        "away_moneyline",
    ]
    consensus = lines.groupby("game_id", as_index=False)[numeric].median()
    latest = lines.groupby("game_id")["retrieved_at"].max()
    consensus["retrieved_at"] = consensus["game_id"].map(latest)
    consensus["sportsbook"] = "consensus"
    return consensus.loc[:, MARKET_COLUMNS]
