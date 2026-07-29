"""Leakage-safe construction of model features for upcoming matchups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.processing import calculate_shannon_entropy, get_db_connection


SCHEDULE_COLUMNS = ["game_id", "season", "week", "home_team", "away_team"]


def _ratio(history: pd.DataFrame, numerator: str, denominator: str) -> float:
    top = history[numerator].sum()
    bottom = history[denominator].sum()
    return float(top / bottom) if bottom else 0.0


def _mean(history: pd.DataFrame, column: str) -> float:
    value = history[column].mean()
    return float(value) if pd.notna(value) else 0.0


def _entropy(history: pd.DataFrame, column: str) -> float:
    values = [str(value) for value in history[column].dropna() if str(value)]
    return calculate_shannon_entropy("|".join(values)) if values else 0.0


def team_state(history: pd.DataFrame) -> dict[str, float]:
    """Aggregate exactly five completed team-games using training formulas."""
    if len(history) != 5:
        raise ValueError(f"Expected five completed games, received {len(history)}")

    state = {
        "avg_offense_points": _mean(history, "team_offense_points"),
        "avg_offense_points_allowed": _mean(history, "opponent_offense_points"),
        "avg_offense_touchdowns": _mean(history, "team_offense_touchdowns"),
        "avg_offense_touchdowns_allowed": _mean(
            history, "opponent_offense_touchdowns"
        ),
        "avg_field_goals": _mean(history, "team_field_goals"),
        "avg_field_goals_allowed": _mean(history, "opponent_field_goals"),
        "avg_points_above_spread": _mean(history, "team_points_above_spread"),
        "avg_result_plus_spread": _mean(history, "team_result_plus_spread"),
        "avg_epa": _ratio(history, "team_sum_epa", "team_count_epa"),
        "avg_redzone_epa": _ratio(
            history, "team_sum_redzone_epa", "team_count_redzone_epa"
        ),
        "avg_nonredzone_epa": _ratio(
            history, "team_sum_nonredzone_epa", "team_count_nonredzone_epa"
        ),
        "avg_pass_epa": _ratio(
            history, "team_sum_pass_epa", "team_count_pass_epa"
        ),
        "avg_rush_epa": _ratio(
            history, "team_sum_rush_epa", "team_count_rush_plays"
        ),
        "avg_success_rate": _ratio(
            history, "team_sum_success", "team_count_success"
        ),
        "cp": _ratio(history, "team_sum_cp", "team_count_cp"),
        "avg_cpoe": _ratio(history, "team_sum_cpoe", "team_count_cpoe"),
        "pass_rate": _ratio(
            history, "team_sum_pass_plays", "team_count_scrimmage_plays"
        ),
        "pass_rate_oe": _ratio(
            history, "team_sum_pass_rate_oe", "team_count_pass_rate_oe"
        ),
        "time_to_throw": _ratio(
            history, "team_sum_time_to_throw", "team_count_time_to_throw"
        ),
        "avg_yac": _ratio(history, "team_sum_yac", "team_count_yac"),
        "avg_air_yards": _ratio(
            history, "team_sum_air_yards", "team_count_air_yards"
        ),
        "avg_yards_pr": _ratio(
            history, "team_sum_rush_yards", "team_count_rush_plays"
        ),
        "avg_yards_pp": _ratio(
            history, "team_sum_pass_yards", "team_count_pass_plays"
        ),
        "avg_rush_plays": float(history["team_count_rush_plays"].sum() / 5),
        "avg_pass_plays": float(history["team_count_pass_plays"].sum() / 5),
        "avg_pressure_rate": _ratio(
            history, "team_sum_pressure", "team_count_pressure"
        ),
        "qb_hit_rate": _ratio(
            history, "team_sum_qb_hit", "team_count_qb_hit"
        ),
        "blitz_rate": _ratio(history, "team_sum_blitz", "team_count_blitz"),
        "avg_defenders_in_box": _ratio(
            history,
            "team_sum_defenders_in_box",
            "team_count_defenders_in_box",
        ),
        "zone_rate": _ratio(history, "team_sum_zone", "team_count_zone"),
        "shotgun_spread_rate": _ratio(
            history,
            "team_sum_shotgun_spread",
            "team_count_shotgun_spread",
        ),
        "heavy_formation_rate": _ratio(
            history,
            "team_sum_heavy_formation",
            "team_count_heavy_formation",
        ),
        "opp_avg_epa": _ratio(
            history, "opponent_sum_epa", "opponent_count_epa"
        ),
        "opp_avg_pass_epa": _ratio(
            history,
            "opponent_sum_pass_epa",
            "opponent_count_pass_epa",
        ),
        "opp_avg_rush_epa": _ratio(
            history,
            "opponent_sum_rush_epa",
            "opponent_count_rush_plays",
        ),
        "opp_avg_success_rate": _ratio(
            history,
            "opponent_sum_success",
            "opponent_count_success",
        ),
        "opp_avg_yards_pr": _ratio(
            history,
            "opponent_sum_rush_yards",
            "opponent_count_rush_plays",
        ),
        "opp_avg_yards_pp": _ratio(
            history,
            "opponent_sum_pass_yards",
            "opponent_count_pass_plays",
        ),
        "opp_avg_pressure_rate": _ratio(
            history,
            "opponent_sum_pressure",
            "opponent_count_pressure",
        ),
        "opp_avg_rush_plays": float(
            history["opponent_count_rush_plays"].sum() / 5
        ),
        "opp_avg_pass_plays": float(
            history["opponent_count_pass_plays"].sum() / 5
        ),
        "offense_entropy": _entropy(history, "team_offense_lineups"),
        "defense_entropy": _entropy(history, "team_defense_lineups"),
    }
    return state


def _matchup_row(game: pd.Series, home: dict, away: dict) -> dict:
    row = {
        "season": int(game["season"]),
        "week": int(game["week"]),
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "game_wind": float(game.get("game_wind", 0) or 0),
        "div_game": int(game.get("div_game", 0) or 0),
    }

    direct = [
        "avg_offense_points",
        "avg_offense_points_allowed",
        "avg_pass_epa",
        "opp_avg_pass_epa",
        "avg_rush_epa",
        "opp_avg_rush_epa",
        "avg_yards_pr",
        "opp_avg_yards_pr",
        "avg_yards_pp",
        "opp_avg_yards_pp",
        "avg_success_rate",
        "opp_avg_success_rate",
        "avg_pressure_rate",
        "opp_avg_pressure_rate",
        "avg_rush_plays",
        "opp_avg_rush_plays",
        "avg_pass_plays",
        "opp_avg_pass_plays",
        "avg_epa",
        "opp_avg_epa",
        "avg_redzone_epa",
        "avg_nonredzone_epa",
        "cp",
        "avg_cpoe",
        "pass_rate",
        "pass_rate_oe",
        "time_to_throw",
        "avg_yac",
        "avg_air_yards",
        "qb_hit_rate",
        "blitz_rate",
        "avg_defenders_in_box",
        "zone_rate",
        "shotgun_spread_rate",
        "heavy_formation_rate",
        "offense_entropy",
        "defense_entropy",
    ]
    for name in direct:
        row[f"home_{name}"] = home[name]
        row[f"away_{name}"] = away[name]

    row.update(
        {
            "diff_home_avg_offense_touchdowns": (
                home["avg_offense_touchdowns"]
                - home["avg_offense_touchdowns_allowed"]
            ),
            "diff_home_avg_field_goals": (
                home["avg_field_goals"] - home["avg_field_goals_allowed"]
            ),
            "diff_away_avg_offense_touchdowns": (
                away["avg_offense_touchdowns"]
                - away["avg_offense_touchdowns_allowed"]
            ),
            "diff_away_avg_field_goals": (
                away["avg_field_goals"] - away["avg_field_goals_allowed"]
            ),
            "home_diff_offensive_points": (
                home["avg_offense_points"]
                - home["avg_offense_points_allowed"]
            ),
            "away_diff_offensive_points": (
                away["avg_offense_points"]
                - away["avg_offense_points_allowed"]
            ),
            "diff_avg_points_above_spread": (
                home["avg_points_above_spread"]
                - away["avg_points_above_spread"]
            ),
            "home_avg_points_above_spread": home["avg_points_above_spread"],
            "away_avg_points_above_spread": away["avg_points_above_spread"],
            "diff_avg_result_plus_spread": (
                home["avg_result_plus_spread"]
                - away["avg_result_plus_spread"]
            ),
            "home_avg_result_plus_spread": home["avg_result_plus_spread"],
            "away_avg_result_plus_spread": away["avg_result_plus_spread"],
        }
    )
    return row


def build_pregame_features(
    schedule: pd.DataFrame,
    *,
    db_path=None,
    required_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build upcoming features and return unavailable games with reasons."""
    missing = sorted(set(SCHEDULE_COLUMNS).difference(schedule.columns))
    if missing:
        raise ValueError(f"Schedule is missing columns: {missing}")
    if schedule["game_id"].duplicated().any():
        raise ValueError("Schedule contains duplicate game IDs")

    with get_db_connection(db_path, read_only=True) as conn:
        history = pd.read_sql_query("SELECT * FROM nfl_data_4", conn)

    available_rows = []
    unavailable_rows = []
    for _, game in schedule.iterrows():
        states = {}
        missing_teams = []
        for side in ["home", "away"]:
            team = game[f"{side}_team"]
            prior = history.loc[
                (history["season"] == game["season"])
                & (history["team"] == team)
                & (history["week"] < game["week"])
            ].sort_values("week").tail(5)
            if len(prior) < 5:
                missing_teams.append(f"{team} ({len(prior)}/5 games)")
            else:
                states[side] = team_state(prior)

        if missing_teams:
            unavailable_rows.append(
                {
                    **game[SCHEDULE_COLUMNS].to_dict(),
                    "reason": "Insufficient current-season history: "
                    + ", ".join(missing_teams),
                }
            )
            continue
        available_rows.append(_matchup_row(game, states["home"], states["away"]))

    features = pd.DataFrame(available_rows)
    if not features.empty:
        available_ids = [
            game["game_id"]
            for _, game in schedule.iterrows()
            if not any(
                item["game_id"] == game["game_id"] for item in unavailable_rows
            )
        ]
        features.index = pd.Index(available_ids, name="game_id")
        features = features.replace([np.inf, -np.inf], np.nan)
        if features.isna().any().any():
            bad = sorted(features.columns[features.isna().any()].tolist())
            raise ValueError(f"Pregame features contain null values: {bad}")
        if required_features:
            missing_features = sorted(set(required_features).difference(features))
            if missing_features:
                raise ValueError(
                    f"Pregame builder is missing model features: {missing_features}"
                )
    unavailable = pd.DataFrame(unavailable_rows)
    return features, unavailable
