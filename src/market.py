"""Provider-independent market-line contracts for weekly predictions."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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


class TheOddsApiProvider(MarketDataProvider):
    """Fetch consensus NFL moneylines from The Odds API."""

    URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    TEAM_NAMES = {
        "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
        "Detroit Lions": "DET", "Green Bay Packers": "GB",
        "Houston Texans": "HOU", "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN", "New England Patriots": "NE",
        "New Orleans Saints": "NO", "New York Giants": "NYG",
        "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Set ODDS_API_KEY before using the live odds provider")

    def get_lines(self, games: pd.DataFrame) -> pd.DataFrame:
        query = urlencode({
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        })
        request = Request(f"{self.URL}?{query}", headers={"User-Agent": "fourth-down-forecast"})
        with urlopen(request, timeout=30) as response:
            events = json.load(response)

        wanted = {
            (row.home_team, row.away_team): str(game_id)
            for game_id, row in games.iterrows()
        }
        rows = []
        for event in events:
            home = self.TEAM_NAMES.get(event.get("home_team"))
            away = self.TEAM_NAMES.get(event.get("away_team"))
            game_id = wanted.get((home, away))
            if game_id is None:
                continue
            prices = []
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "h2h":
                        continue
                    outcomes = {item["name"]: item["price"] for item in market.get("outcomes", [])}
                    if event.get("home_team") in outcomes and event.get("away_team") in outcomes:
                        prices.append((outcomes[event["home_team"]], outcomes[event["away_team"]]))
            if prices:
                rows.append({
                    "game_id": game_id,
                    "sportsbook": "consensus",
                    "market_home_margin": None,
                    "home_spread_price": None,
                    "away_spread_price": None,
                    "total": None,
                    "home_moneyline": float(pd.Series([p[0] for p in prices]).median()),
                    "away_moneyline": float(pd.Series([p[1] for p in prices]).median()),
                    "retrieved_at": pd.Timestamp.now(tz="UTC"),
                })
        return validate_market_lines(pd.DataFrame(rows, columns=MARKET_COLUMNS))


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
