import math
import sqlite3
from collections import Counter
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl.db"
RAW_PLAYS_TABLE = "nfl_data_1"
FINAL_FEATURE_VIEW = "nfl_data_8"


def _resolve_db_path(db_path=None):
    """Return an absolute database path and ensure its parent exists."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def data_collection_to_sql(seasons=range(2015, 2026), db_path=None):
    """Download play-by-play data and atomically replace the raw plays table."""
    seasons = sorted({int(season) for season in seasons})
    if not seasons:
        raise ValueError("At least one season is required.")

    df = nfl.import_pbp_data(seasons)
    if df.empty:
        raise RuntimeError("The play-by-play download returned no rows.")

    staging_table = f"{RAW_PLAYS_TABLE}_staging"
    with get_db_connection(db_path) as conn:
        df.to_sql(staging_table, conn, if_exists="replace", index=False)
        conn.execute(f'DROP TABLE IF EXISTS "{RAW_PLAYS_TABLE}"')
        conn.execute(f'ALTER TABLE "{staging_table}" RENAME TO "{RAW_PLAYS_TABLE}"')


def get_db_connection(db_path=None, *, read_only=False):
    """Create a consistently configured SQLite connection."""
    path = _resolve_db_path(db_path)
    if read_only:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=60)
    else:
        conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def _drop_object(conn, name):
    """Drop a SQLite table or view without assuming which type currently exists."""
    row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = ?", (name,)
    ).fetchone()
    if row is not None:
        object_type = row[0].upper()
        if object_type not in {"TABLE", "VIEW"}:
            raise RuntimeError(f"Cannot replace unsupported SQLite object: {name}")
        conn.execute(f'DROP {object_type} "{name}"')


def cut_columns():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS nfl_data_2")

    query = """
    CREATE VIEW nfl_data_2 AS 
    SELECT 
        -- Structural Metadata
        game_id,
        season,
        week,
        div_game,                   -- Divisional rivalry flag (1 or 0)
        home_team,
        away_team,
        posteam,
        defteam,
        posteam_type,               -- 'home' or 'away'
        season_type,
        
        -- Game State & Live Scoring
        qtr,
        down,
        ydstogo,
        yardline_100,               -- Distance to opponent goal line
        game_seconds_remaining,
        home_score,                 -- Current home score
        away_score,                 -- Current away score
        score_differential,         -- Live margin spread
        spread_line,                -- Pre-game closing Vegas line
        
        -- Play Identification & Basic Outputs
        play_id,
        play_type,                  -- Restricted to 'pass' or 'run' via WHERE clause
        yards_gained,
        yards_after_catch,          -- Target receiver YAC (Pass only)
        air_yards,
        complete_pass,
        time_to_throw,              -- NGS pass release timing (Pass only)
        pass_touchdown,
        rush_touchdown,
        field_goal_result,
        interception,
        sack,
        fumble,
        
        -- Efficiency & Advanced Machine Learning Metrics
        epa,                        -- Expected Points Added
        qb_epa,                     -- QB-specific EPA
        success,                    -- Binary play success (1 or 0)
        cpoe,                       -- Completion Percentage Over Expected
        cp,
        xpass,                      -- Expected Pass Probability
        pass_oe,                    -- Pass Over Expected
        
        -- Advanced Defensive Tracking Variables
        was_pressure,               -- Binary flag if QB was pressured
        qb_hit,
        number_of_pass_rushers,
        defenders_in_box,
        defense_coverage_type,      -- Premium coverage structure tracking
        defense_man_zone_type,     -- Premium scheme split tracking
        
        -- Roster Continuity & Personnel Strings
        offense_personnel,          -- e.g., '11 personnel'
        defense_personnel,
        offense_formation,
        offense_players,            -- String of the 11 offensive player IDs on field
        defense_players,            -- String of the 11 defensive player IDs on field
        
        -- Market & Situation Probabilities
        wp,                         -- Raw win probability
        vegas_wp,                   -- Vegas market-calibrated win probability
        
        -- Environmental Factors
        roof,
        surface,
        temp,
        wind
        
    FROM nfl_data_1 
    """

    cursor.execute(query)
    conn.commit()
    conn.close()
    print("cut columns")


def make_game_features():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS nfl_data_3")

    query = """
    CREATE VIEW nfl_data_3 AS 
    SELECT
    game_id,
    season,
    week,
    home_team,
    away_team,
    div_game,
    season_type,
    MAX(home_score) AS home_score_final,
    MAX(away_score) AS away_score_final,
    MAX(home_score) - MAX(away_score) AS home_final_spread,
    (MAX(home_score) - MAX(away_score)) - MAX(spread_line) AS home_points_above_spread,
    (MAX(away_score) - MAX(home_score)) + MAX(spread_line) AS away_points_above_spread,
    (MAX(home_score) - MAX(away_score)) + MAX(spread_line) AS home_result_plus_spread,
    (MAX(away_score) - MAX(home_score)) - MAX(spread_line) AS away_result_plus_spread,
    MAX(spread_line) AS spread_line,
    MAX(CASE WHEN roof IN ('dome','closed') THEN 1 ELSE 0 END) AS roof_adjusted,
    MAX(CASE WHEN roof IN ('dome','closed') THEN 70 ELSE temp END) AS temp_adjusted,
    MAX(CASE WHEN roof IN ('dome','closed') THEN 0 ELSE wind END) AS wind_adjusted,
    
    -- Offensive Points --
    COALESCE(SUM(CASE WHEN posteam_type = 'home' AND field_goal_result = 'made' THEN 1 END),0) * 3 + COALESCE(SUM(CASE WHEN posteam_type = 'home' THEN COALESCE(pass_touchdown,0) + COALESCE(rush_touchdown,0) END),0) * 7 AS home_offense_points,
    COALESCE(SUM(CASE WHEN posteam_type = 'away' AND field_goal_result = 'made' THEN 1 END),0) * 3 + COALESCE(SUM(CASE WHEN posteam_type = 'away' THEN COALESCE(pass_touchdown,0) + COALESCE(rush_touchdown,0) END),0) * 7 AS away_offense_points,

    -- Offensive Touchdowns --
    COALESCE(SUM(CASE WHEN posteam_type = 'home' THEN COALESCE(pass_touchdown,0) + COALESCE(rush_touchdown,0) END),0) AS home_offense_touchdowns,
    COALESCE(SUM(CASE WHEN posteam_type = 'away' THEN COALESCE(pass_touchdown,0) + COALESCE(rush_touchdown,0) END),0) AS away_offense_touchdowns,

    -- Field Goals --
    COALESCE(SUM(CASE WHEN posteam_type = 'home' AND field_goal_result = 'made' THEN 1 END),0) AS home_field_goals,
    COALESCE(SUM(CASE WHEN posteam_type = 'away' AND field_goal_result = 'made' THEN 1 END),0) AS away_field_goals,
    
    -- EPA Total Scrimmage
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN epa END) AS home_sum_epa,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS home_count_epa,
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN epa END) AS away_sum_epa,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS away_count_epa,
    
    -- Redzone EPA
    SUM(CASE WHEN posteam_type = 'home' AND yardline_100 <= 20 AND play_type IN ('pass','run') THEN epa END) AS home_sum_redzone_epa,
    COUNT(CASE WHEN posteam_type = 'home' AND yardline_100 <= 20 AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS home_count_redzone_epa,
    SUM(CASE WHEN posteam_type = 'away' AND yardline_100 <= 20 AND play_type IN ('pass','run') THEN epa END) AS away_sum_redzone_epa,
    COUNT(CASE WHEN posteam_type = 'away' AND yardline_100 <= 20 AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS away_count_redzone_epa,
    
    -- Non-Redzone EPA
    SUM(CASE WHEN posteam_type = 'home' AND yardline_100 > 20 AND play_type IN ('pass','run') THEN epa END) AS home_sum_nonredzone_epa,
    COUNT(CASE WHEN posteam_type = 'home' AND yardline_100 > 20 AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS home_count_nonredzone_epa,
    SUM(CASE WHEN posteam_type = 'away' AND yardline_100 > 20 AND play_type IN ('pass','run') THEN epa END) AS away_sum_nonredzone_epa,
    COUNT(CASE WHEN posteam_type = 'away' AND yardline_100 > 20 AND play_type IN ('pass','run') AND epa IS NOT NULL THEN 1 END) AS away_count_nonredzone_epa,
    
    -- Passing EPA
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN epa END) AS home_sum_pass_epa,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND epa IS NOT NULL THEN 1 END) AS home_count_pass_epa,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN epa END) AS away_sum_pass_epa,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND epa IS NOT NULL THEN 1 END) AS away_count_pass_epa,

    -- Rushing EPA
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'run' THEN epa END) AS home_sum_rush_epa,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'run' AND epa IS NOT NULL THEN 1 END) AS home_count_rush_epa,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'run' THEN epa END) AS away_sum_rush_epa,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'run' AND epa IS NOT NULL THEN 1 END) AS away_count_rush_epa,
    
    -- Success Rate
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN success END) AS home_sum_success,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND success IS NOT NULL THEN 1 END) AS home_count_success,
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN success END) AS away_sum_success,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND success IS NOT NULL THEN 1 END) AS away_count_success,
    
    -- Completion Percentage (CP)
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN complete_pass END) AS home_sum_cp,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND complete_pass IS NOT NULL THEN 1 END) AS home_count_cp,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN complete_pass END) AS away_sum_cp,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND complete_pass IS NOT NULL THEN 1 END) AS away_count_cp,
    
    -- CPOE
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN cpoe END) AS home_sum_cpoe,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND cpoe IS NOT NULL THEN 1 END) AS home_count_cpoe,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN cpoe END) AS away_sum_cpoe,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND cpoe IS NOT NULL THEN 1 END) AS away_count_cpoe,
    
    -- Pass and Rush Rate 
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN 1 ELSE 0 END) AS home_sum_pass_plays,
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'run' THEN 1 ELSE 0 END) AS home_sum_rush_plays,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN 1 END) AS home_count_scrimmage_plays,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN 1 ELSE 0 END) AS away_sum_pass_plays,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'run' THEN 1 ELSE 0 END) AS away_sum_rush_plays,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN 1 END) AS away_count_scrimmage_plays,
    
    -- Pass OE
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN pass_oe END) AS home_sum_pass_rate_oe,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND pass_oe IS NOT NULL THEN 1 END) AS home_count_pass_rate_oe,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN pass_oe END) AS away_sum_pass_rate_oe,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND pass_oe IS NOT NULL THEN 1 END) AS away_count_pass_rate_oe,
    
    -- Time to Throw
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN time_to_throw END) AS home_sum_time_to_throw,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND time_to_throw IS NOT NULL THEN 1 END) AS home_count_time_to_throw,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN time_to_throw END) AS away_sum_time_to_throw,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND time_to_throw IS NOT NULL THEN 1 END) AS away_count_time_to_throw,
    
    -- Yards After Catch (YAC)
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND complete_pass = 1 THEN yards_after_catch END) AS home_sum_yac,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND complete_pass = 1 AND yards_after_catch IS NOT NULL THEN 1 END) AS home_count_yac,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND complete_pass = 1 THEN yards_after_catch END) AS away_sum_yac,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND complete_pass = 1 AND yards_after_catch IS NOT NULL THEN 1 END) AS away_count_yac,
    
    -- Air Yards
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND complete_pass = 1 THEN yards_gained - yards_after_catch END) AS home_sum_air_yards,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND complete_pass = 1 AND yards_gained IS NOT NULL AND yards_after_catch IS NOT NULL THEN 1 END) AS home_count_air_yards,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND complete_pass = 1 THEN yards_gained - yards_after_catch END) AS away_sum_air_yards,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND complete_pass = 1 AND yards_gained IS NOT NULL AND yards_after_catch IS NOT NULL THEN 1 END) AS away_count_air_yards,
    
    -- Yards Per Pass
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN yards_gained END) AS home_sum_pass_yards,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND yards_gained IS NOT NULL THEN 1 END) AS home_count_pass_plays,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN yards_gained END) AS away_sum_pass_yards,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND yards_gained IS NOT NULL THEN 1 END) AS away_count_pass_plays,
        
    -- Yards Per Rush
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'run' THEN yards_gained END) AS home_sum_rush_yards,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'run' AND yards_gained IS NOT NULL THEN 1 END) AS home_count_rush_plays,
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'run' THEN yards_gained END) AS away_sum_rush_yards,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'run' AND yards_gained IS NOT NULL THEN 1 END) AS away_count_rush_plays,
    
    -- Pressure Rate
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN was_pressure END) AS home_sum_pressure,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND was_pressure IS NOT NULL THEN 1 END) AS home_count_pressure,
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN was_pressure END) AS away_sum_pressure,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND was_pressure IS NOT NULL THEN 1 END) AS away_count_pressure,
    
    -- QB Hit Rate
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN qb_hit END) AS home_sum_qb_hit,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND qb_hit IS NOT NULL THEN 1 END) AS home_count_qb_hit,
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN qb_hit END) AS away_sum_qb_hit,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND qb_hit IS NOT NULL THEN 1 END) AS away_count_qb_hit,
    
    -- Blitz Rate
    SUM(CASE WHEN posteam_type = 'away' AND play_type = 'pass' THEN CASE WHEN number_of_pass_rushers > 4 THEN 1 ELSE 0 END END) AS home_sum_blitz,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type = 'pass' AND number_of_pass_rushers IS NOT NULL THEN 1 END) AS home_count_blitz,
    SUM(CASE WHEN posteam_type = 'home' AND play_type = 'pass' THEN CASE WHEN number_of_pass_rushers > 4 THEN 1 ELSE 0 END END) AS away_sum_blitz,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type = 'pass' AND number_of_pass_rushers IS NOT NULL THEN 1 END) AS away_count_blitz,
    
    -- Defenders in Box
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN defenders_in_box END) AS home_sum_defenders_in_box,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND defenders_in_box IS NOT NULL THEN 1 END) AS home_count_defenders_in_box,
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN defenders_in_box END) AS away_sum_defenders_in_box,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND defenders_in_box IS NOT NULL THEN 1 END) AS away_count_defenders_in_box,
    
    -- Zone Rate
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND defense_man_zone_type = 'ZONE_COVERAGE' THEN 1 ELSE 0 END) AS home_sum_zone,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND defense_man_zone_type IN ('ZONE_COVERAGE', 'MAN_COVERAGE') THEN 1 END) AS home_count_zone,
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND defense_man_zone_type = 'ZONE_COVERAGE' THEN 1 ELSE 0 END) AS away_sum_zone,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND defense_man_zone_type IN ('ZONE_COVERAGE', 'MAN_COVERAGE') THEN 1 END) AS away_count_zone,
    
    -- Shotgun/Spread Rate
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND offense_formation IN ('SHOTGUN', 'EMPTY') THEN 1.0 ELSE 0.0 END) AS home_sum_shotgun_spread,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND offense_formation IS NOT NULL THEN 1 END) AS home_count_shotgun_spread,
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND offense_formation IN ('SHOTGUN', 'EMPTY') THEN 1.0 ELSE 0.0 END) AS away_sum_shotgun_spread,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND offense_formation IS NOT NULL THEN 1 END) AS away_count_shotgun_spread,
    
    -- Heavy Formation Rate
    SUM(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND offense_formation IN ('SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'JUMBO') THEN 1.0 ELSE 0.0 END) AS home_sum_heavy_formation,
    COUNT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') AND offense_formation IS NOT NULL THEN 1 END) AS home_count_heavy_formation,
    SUM(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND offense_formation IN ('SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'JUMBO') THEN 1.0 ELSE 0.0 END) AS away_sum_heavy_formation,
    COUNT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') AND offense_formation IS NOT NULL THEN 1 END) AS away_count_heavy_formation,
    
    -- Lineup Raw Strings
    GROUP_CONCAT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN offense_players END, '|') AS home_offense_lineups,
    GROUP_CONCAT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN defense_players END, '|') AS home_defense_lineups,
    GROUP_CONCAT(CASE WHEN posteam_type = 'away' AND play_type IN ('pass','run') THEN offense_players END, '|') AS away_offense_lineups,
    GROUP_CONCAT(CASE WHEN posteam_type = 'home' AND play_type IN ('pass','run') THEN defense_players END, '|') AS away_defense_lineups
    
    FROM nfl_data_2 GROUP BY game_id"""

    cursor.execute(query)
    conn.commit()
    conn.close()
    print("make game features complete")


def order_games():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS nfl_data_4")
    query = """
    CREATE VIEW nfl_data_4 AS 
    SELECT
    game_id,
    1 AS is_home,
    season,
    week,
    season_type,
    home_team AS team,
    away_team AS opponent,
    div_game,
    home_score_final AS team_score,
    away_score_final AS opponent_score,
    home_points_above_spread AS team_points_above_spread,
    away_points_above_spread AS opponent_points_above_spread,
    home_result_plus_spread AS team_result_plus_spread,
    away_result_plus_spread AS opponent_result_plus_spread,
    spread_line,
    roof_adjusted,
    temp_adjusted,
    wind_adjusted,
    
    home_offense_points AS team_offense_points,
    home_offense_touchdowns AS team_offense_touchdowns,
    home_field_goals AS team_field_goals,
    away_offense_points AS opponent_offense_points,
    away_offense_touchdowns AS opponent_offense_touchdowns,
    away_field_goals AS opponent_field_goals,

    

    
    -- Team Sums & Counts
    home_sum_epa AS team_sum_epa, home_count_epa AS team_count_epa,
    home_sum_redzone_epa AS team_sum_redzone_epa, home_count_redzone_epa AS team_count_redzone_epa,
    home_sum_nonredzone_epa AS team_sum_nonredzone_epa, home_count_nonredzone_epa AS team_count_nonredzone_epa,
    home_sum_pass_epa AS team_sum_pass_epa, home_count_pass_epa AS team_count_pass_epa,
    home_sum_rush_epa AS team_sum_rush_epa, home_count_rush_epa AS team_count_rush_epa,
    home_sum_success AS team_sum_success, home_count_success AS team_count_success,
    home_sum_cp AS team_sum_cp, home_count_cp AS team_count_cp,
    home_sum_cpoe AS team_sum_cpoe, home_count_cpoe AS team_count_cpoe,
    home_sum_pass_plays AS team_sum_pass_plays, home_count_scrimmage_plays AS team_count_scrimmage_plays,
    home_sum_rush_plays AS team_sum_rush_plays,
    home_sum_pass_rate_oe AS team_sum_pass_rate_oe, home_count_pass_rate_oe AS team_count_pass_rate_oe,
    home_sum_time_to_throw AS team_sum_time_to_throw, home_count_time_to_throw AS team_count_time_to_throw,
    home_sum_yac AS team_sum_yac, home_count_yac AS team_count_yac,
    home_sum_air_yards AS team_sum_air_yards, home_count_air_yards AS team_count_air_yards,
    home_sum_pass_yards AS team_sum_pass_yards, home_count_pass_plays AS team_count_pass_plays,
    home_sum_rush_yards AS team_sum_rush_yards, home_count_rush_plays AS team_count_rush_plays,
    home_sum_pressure AS team_sum_pressure, home_count_pressure AS team_count_pressure,
    home_sum_qb_hit AS team_sum_qb_hit, home_count_qb_hit AS team_count_qb_hit,
    home_sum_blitz AS team_sum_blitz, home_count_blitz AS team_count_blitz,
    home_sum_defenders_in_box AS team_sum_defenders_in_box, home_count_defenders_in_box AS team_count_defenders_in_box,
    home_sum_zone AS team_sum_zone, home_count_zone AS team_count_zone,
    home_sum_shotgun_spread AS team_sum_shotgun_spread, home_count_shotgun_spread AS team_count_shotgun_spread,
    home_sum_heavy_formation AS team_sum_heavy_formation, home_count_heavy_formation AS team_count_heavy_formation,
    
    -- Raw Strings for Lineup Entropy
    home_offense_lineups AS team_offense_lineups,
    home_defense_lineups AS team_defense_lineups,
    
    -- Opponent Performance Metrics (Sums & Counts)
    away_sum_epa AS opponent_sum_epa, away_count_epa AS opponent_count_epa,
    away_sum_pass_epa AS opponent_sum_pass_epa, away_count_pass_epa AS opponent_count_pass_epa,
    away_sum_rush_epa AS opponent_sum_rush_epa, away_count_rush_epa AS opponent_count_rush_epa,
    away_sum_pass_yards AS opponent_sum_pass_yards, away_count_pass_plays AS opponent_count_pass_plays,
    away_sum_rush_yards AS opponent_sum_rush_yards, away_count_rush_plays AS opponent_count_rush_plays,
    away_sum_success AS opponent_sum_success, away_count_success AS opponent_count_success,
    away_sum_time_to_throw AS opponent_sum_time_to_throw, away_count_time_to_throw AS opponent_count_time_to_throw,
    away_sum_pressure AS opponent_sum_pressure, away_count_pressure AS opponent_count_pressure,
    away_sum_pass_plays AS opponent_sum_pass_plays,
    away_sum_rush_plays AS opponent_sum_rush_plays
    FROM nfl_data_3
    WHERE season_type = 'REG' AND season > 2017

    UNION ALL

    SELECT
    game_id,
    0 AS is_home,
    season,
    week,
    season_type,
    away_team AS team,
    home_team AS opponent,
    div_game,
    away_score_final AS team_score,
    home_score_final AS opponent_score,
    away_points_above_spread AS team_points_above_spread,
    home_points_above_spread AS opponent_points_above_spread,
    away_result_plus_spread AS team_result_plus_spread,
    home_result_plus_spread AS opponent_result_plus_spread,
    spread_line,
    roof_adjusted,
    temp_adjusted,
    wind_adjusted,
    
    away_offense_points AS team_offense_points,
    away_offense_touchdowns AS team_offense_touchdowns,
    away_field_goals AS team_field_goals,
    home_offense_points AS opponent_offense_points,
    home_offense_touchdowns AS opponent_offense_touchdowns,
    home_field_goals AS opponent_field_goals,
    
    -- Team Sums & Counts
    away_sum_epa AS team_sum_epa, away_count_epa AS team_count_epa,
    away_sum_redzone_epa AS team_sum_redzone_epa, away_count_redzone_epa AS team_count_redzone_epa,
    away_sum_nonredzone_epa AS team_sum_nonredzone_epa, away_count_nonredzone_epa AS team_count_nonredzone_epa,
    away_sum_pass_epa AS team_sum_pass_epa, away_count_pass_epa AS team_count_pass_epa,
    away_sum_rush_epa AS team_sum_rush_epa, away_count_rush_epa AS team_count_rush_epa,
    away_sum_success AS team_sum_success, away_count_success AS team_count_success,
    away_sum_cp AS team_sum_cp, away_count_cp AS team_count_cp,
    away_sum_cpoe AS team_sum_cpoe, away_count_cpoe AS team_count_cpoe,
    away_sum_pass_plays AS team_sum_pass_plays, away_count_scrimmage_plays AS team_count_scrimmage_plays,
    away_sum_rush_plays AS team_sum_rush_plays,
    away_sum_pass_rate_oe AS team_sum_pass_rate_oe, away_count_pass_rate_oe AS team_count_pass_rate_oe,
    away_sum_time_to_throw AS team_sum_time_to_throw, away_count_time_to_throw AS team_count_time_to_throw,
    away_sum_yac AS team_sum_yac, away_count_yac AS team_count_yac,
    away_sum_air_yards AS team_sum_air_yards, away_count_air_yards AS team_count_air_yards,
    away_sum_pass_yards AS team_sum_pass_yards, away_count_pass_plays AS team_count_pass_plays,
    away_sum_rush_yards AS team_sum_rush_yards, away_count_rush_plays AS team_count_rush_plays,
    away_sum_pressure AS team_sum_pressure, away_count_pressure AS team_count_pressure,
    away_sum_qb_hit AS team_sum_qb_hit, away_count_qb_hit AS team_count_qb_hit,
    away_sum_blitz AS team_sum_blitz, away_count_blitz AS team_count_blitz,
    away_sum_defenders_in_box AS team_sum_defenders_in_box, away_count_defenders_in_box AS team_count_defenders_in_box,
    away_sum_zone AS team_sum_zone, away_count_zone AS team_count_zone,
    away_sum_shotgun_spread AS team_sum_shotgun_spread, away_count_shotgun_spread AS team_count_shotgun_spread,
    away_sum_heavy_formation AS team_sum_heavy_formation, away_count_heavy_formation AS team_count_heavy_formation,
    
    -- Raw Strings for Lineup Entropy
    away_offense_lineups AS team_offense_lineups,
    away_defense_lineups AS team_defense_lineups,
    
    -- Opponent Performance Metrics (Sums & Counts)
    home_sum_epa AS opponent_sum_epa, home_count_epa AS opponent_count_epa,
    home_sum_pass_epa AS opponent_sum_pass_epa, home_count_pass_epa AS opponent_count_pass_epa,
    home_sum_rush_epa AS opponent_sum_rush_epa, home_count_rush_epa AS opponent_count_rush_epa,
    home_sum_pass_yards AS opponent_sum_pass_yards, home_count_pass_plays AS opponent_count_pass_plays,
    home_sum_rush_yards AS opponent_sum_rush_yards, home_count_rush_plays AS opponent_count_rush_plays,
    home_sum_success AS opponent_sum_success, home_count_success AS opponent_count_success,
    home_sum_time_to_throw AS opponent_sum_time_to_throw, home_count_time_to_throw AS opponent_count_time_to_throw,
    home_sum_pressure AS opponent_sum_pressure, home_count_pressure AS opponent_count_pressure,
    home_sum_pass_plays AS opponent_sum_pass_plays,
    home_sum_rush_plays AS opponent_sum_rush_plays
    FROM nfl_data_3
    WHERE season_type = 'REG' AND season > 2017
    
    ORDER BY team, season, week;
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    print("order games pipeline aligned")


def make_rolling_features():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DROP VIEW IF EXISTS nfl_data_5")
    query = """
    CREATE VIEW nfl_data_5 AS
    WITH rolling_sums AS (
        SELECT 
            game_id,
            is_home,
            season,
            week,
            team,
            opponent,
            div_game,
            team_score,
            opponent_score,
            spread_line,
            team_result_plus_spread,
            opponent_result_plus_spread,
            team_offense_points,
            team_offense_touchdowns,
            team_field_goals,
            opponent_offense_points,
            opponent_offense_touchdowns,
            opponent_field_goals,
            roof_adjusted,
            temp_adjusted,
            wind_adjusted,
            CASE WHEN team_count_pass_plays > 0 THEN CAST(team_sum_pass_yards AS REAL) / team_count_pass_plays ELSE 0.0 END AS team_pass_ypp,
            CASE WHEN team_count_rush_plays > 0 THEN CAST(team_sum_rush_yards AS REAL) / team_count_rush_plays ELSE 0.0 END AS team_rush_ypr,
            CASE WHEN opponent_count_pass_plays > 0 THEN CAST(opponent_sum_pass_yards AS REAL) / opponent_count_pass_plays ELSE 0.0 END AS opponent_pass_yards_pp,
            CASE WHEN opponent_count_rush_plays > 0 THEN CAST(opponent_sum_rush_yards AS REAL) / opponent_count_rush_plays ELSE 0.0 END AS opponent_rush_yards_pr,
            CAST(team_sum_pass_epa AS REAL) / team_count_pass_epa AS team_pass_epa_pp,
            CAST(team_sum_rush_epa AS REAL) / team_count_rush_epa AS team_rush_epa_pr,
            CAST(opponent_sum_pass_epa AS REAL) / opponent_count_pass_epa AS opponent_pass_epa_pp,
            CAST(opponent_sum_rush_epa AS REAL) / opponent_count_rush_epa AS opponent_rush_epa_pr,
            CASE WHEN team_count_pressure > 0 THEN CAST(COALESCE(team_sum_pressure, 0) AS REAL) / team_count_pressure ELSE 0.0 END AS team_pressure_rate,
            CASE WHEN opponent_count_pressure > 0 THEN CAST(COALESCE(opponent_sum_pressure, 0) AS REAL) / opponent_count_pressure ELSE 0.0 END AS opponent_pressure_rate,
            team_sum_pass_plays AS team_pass_plays,
            team_sum_rush_plays AS team_rush_plays,
            opponent_sum_pass_plays AS opponent_pass_plays,
            opponent_sum_rush_plays AS opponent_rush_plays,
            CASE WHEN team_count_success > 0 THEN CAST(team_sum_success AS REAL) / team_count_success ELSE 0.0 END AS team_success_rate,
            CASE WHEN opponent_count_success > 0 THEN CAST(opponent_sum_success AS REAL) / opponent_count_success ELSE 0.0 END AS opponent_success_rate,
            
            AVG(team_offense_points) OVER w AS roll5_avg_offense_points,
            AVG(team_offense_touchdowns) OVER w AS roll5_avg_offense_touchdowns,
            AVG(team_field_goals) OVER w AS roll5_avg_field_goals,
            AVG(opponent_offense_points) OVER w AS roll5_avg_offense_points_allowed,
            AVG(opponent_offense_touchdowns) OVER w AS roll5_avg_offense_touchdowns_allowed,
            AVG(opponent_field_goals) OVER w AS roll5_avg_field_goals_allowed,
            AVG(team_points_above_spread) OVER w AS roll5_avg_points_above_spread,
            AVG(team_result_plus_spread) OVER w AS roll5_avg_result_plus_spread,
            
            -- String Aggregation over the 5-game historical window
            GROUP_CONCAT(team_offense_lineups, '|') OVER w AS roll5_team_offense_history,
            GROUP_CONCAT(team_defense_lineups, '|') OVER w AS roll5_team_defense_history,

            -- Rolling sums of numerators
            SUM(team_sum_epa) OVER w AS roll5_sum_epa,
            SUM(team_sum_redzone_epa) OVER w AS roll5_sum_redzone_epa,
            SUM(team_sum_nonredzone_epa) OVER w AS roll5_sum_nonredzone_epa,
            SUM(team_sum_pass_epa) OVER w AS roll5_sum_pass_epa,
            SUM(team_sum_rush_epa) OVER w AS roll5_sum_rush_epa,
            SUM(team_sum_success) OVER w AS roll5_sum_success,
            SUM(team_sum_cp) OVER w AS roll5_sum_cp,
            SUM(team_sum_cpoe) OVER w AS roll5_sum_cpoe,
            SUM(team_sum_pass_plays) OVER w AS roll5_sum_pass_plays,
            SUM(team_sum_rush_plays) OVER w AS roll5_sum_rush_plays,
            SUM(team_sum_pass_rate_oe) OVER w AS roll5_sum_pass_rate_oe,
            SUM(team_sum_time_to_throw) OVER w AS roll5_sum_time_to_throw,
            SUM(team_sum_yac) OVER w AS roll5_sum_yac,
            SUM(team_sum_air_yards) OVER w AS roll5_sum_air_yards,
            SUM(team_sum_pass_yards) OVER w AS roll5_sum_pass_yards,
            SUM(team_sum_rush_yards) OVER w AS roll5_sum_rush_yards,
            SUM(team_sum_pressure) OVER w AS roll5_sum_pressure,
            SUM(team_sum_qb_hit) OVER w AS roll5_sum_qb_hit,
            SUM(team_sum_blitz) OVER w AS roll5_sum_blitz,
            SUM(team_sum_defenders_in_box) OVER w AS roll5_sum_defenders_in_box,
            SUM(team_sum_zone) OVER w AS roll5_sum_zone,
            SUM(team_sum_shotgun_spread) OVER w AS roll5_sum_shotgun_spread,
            SUM(team_sum_heavy_formation) OVER w AS roll5_sum_heavy_formation,
            
            -- Rolling sums of denominators
            SUM(team_count_epa) OVER w AS roll5_count_epa,
            SUM(team_count_redzone_epa) OVER w AS roll5_count_redzone_epa,
            SUM(team_count_nonredzone_epa) OVER w AS roll5_count_nonredzone_epa,
            SUM(team_count_pass_epa) OVER w AS roll5_count_pass_epa,
            SUM(team_count_success) OVER w AS roll5_count_success,
            SUM(team_count_cp) OVER w AS roll5_count_cp,
            SUM(team_count_cpoe) OVER w AS roll5_count_cpoe,
            SUM(team_count_scrimmage_plays) OVER w AS roll5_count_scrimmage_plays,
            SUM(team_count_pass_rate_oe) OVER w AS roll5_count_pass_rate_oe,
            SUM(team_count_time_to_throw) OVER w AS roll5_count_time_to_throw,
            SUM(team_count_yac) OVER w AS roll5_count_yac,
            SUM(team_count_air_yards) OVER w AS roll5_count_air_yards,
            SUM(team_count_rush_plays) OVER w AS roll5_count_rush_plays,
            SUM(team_count_pass_plays) OVER w AS roll5_count_pass_plays,
            SUM(team_count_pressure) OVER w AS roll5_count_pressure,
            SUM(team_count_qb_hit) OVER w AS roll5_count_qb_hit,
            SUM(team_count_blitz) OVER w AS roll5_count_blitz,
            SUM(team_count_defenders_in_box) OVER w AS roll5_count_defenders_in_box,
            SUM(team_count_zone) OVER w AS roll5_count_zone,
            SUM(team_count_shotgun_spread) OVER w AS roll5_count_shotgun_spread,
            SUM(team_count_heavy_formation) OVER w AS roll5_count_heavy_formation,

            -- Opponent Historical Rolling Metrics
            SUM(opponent_sum_epa) OVER w AS roll5_opp_sum_epa,
            SUM(opponent_count_epa) OVER w AS roll5_opp_count_epa,
            SUM(opponent_sum_pass_epa) OVER w AS roll5_opp_sum_pass_epa,
            SUM(opponent_count_pass_epa) OVER w AS roll5_opp_count_pass_epa,
            SUM(opponent_sum_rush_epa) OVER w AS roll5_opp_sum_rush_epa,
            SUM(opponent_count_rush_epa) OVER w AS roll5_opp_count_rush_epa,
            SUM(opponent_sum_success) OVER w AS roll5_opp_sum_success,
            SUM(opponent_count_success) OVER w AS roll5_opp_count_success,
            SUM(opponent_sum_pass_yards) OVER w AS roll5_opp_sum_pass_yards,
            SUM(opponent_count_pass_plays) OVER w AS roll5_opp_count_pass_plays,
            SUM(opponent_sum_rush_yards) OVER w AS roll5_opp_sum_rush_yards,
            SUM(opponent_count_rush_plays) OVER w AS roll5_opp_count_rush_plays,
            SUM(opponent_sum_pressure) OVER w AS roll5_opp_sum_pressure,
            SUM(opponent_count_pressure) OVER w AS roll5_opp_count_pressure,

            COUNT(1) OVER w AS roll5_game_count
            
        FROM nfl_data_4
        WINDOW w AS (
            PARTITION BY team, season
            ORDER BY season, week 
            ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
        )
    )
    SELECT 
        game_id,
            is_home,
            season,
            week,
            team,
            opponent,
            div_game,
            team_score,
            opponent_score,
            spread_line,
            team_result_plus_spread,
            opponent_result_plus_spread,
            team_offense_points,
            team_offense_touchdowns,
            team_field_goals,
            opponent_offense_points,
            opponent_offense_touchdowns,
            opponent_field_goals,
            roof_adjusted,
            temp_adjusted,
            wind_adjusted,
            team_pass_ypp,
            team_rush_ypr,
            opponent_pass_yards_pp,
            opponent_rush_yards_pr,
            team_pass_epa_pp,
            team_rush_epa_pr,
            opponent_pass_epa_pp,
            opponent_rush_epa_pr,
            team_pressure_rate,
            opponent_pressure_rate,
            team_pass_plays,
            team_rush_plays,
            opponent_pass_plays,
            opponent_rush_plays,
            team_success_rate,
            opponent_success_rate,
        roll5_team_offense_history,
        roll5_team_defense_history,
        roll5_avg_offense_points,
        roll5_avg_offense_touchdowns,
        roll5_avg_field_goals,
        roll5_avg_offense_points_allowed,
        roll5_avg_offense_touchdowns_allowed,
        roll5_avg_field_goals_allowed,
        roll5_avg_points_above_spread,
        roll5_avg_result_plus_spread,
        roll5_game_count,
        
        -- True weighted averages (Protected via CAST to prevent integer division flattening)
        CASE WHEN roll5_count_epa > 0 THEN CAST(roll5_sum_epa AS REAL) / roll5_count_epa ELSE 0.0 END AS roll5_avg_epa,
        CASE WHEN roll5_count_redzone_epa > 0 THEN CAST(roll5_sum_redzone_epa AS REAL) / roll5_count_redzone_epa ELSE 0.0 END AS roll5_avg_redzone_epa,
        CASE WHEN roll5_count_nonredzone_epa > 0 THEN CAST(roll5_sum_nonredzone_epa AS REAL) / roll5_count_nonredzone_epa ELSE 0.0 END AS roll5_avg_nonredzone_epa,
        CAST(roll5_sum_pass_epa AS REAL) / roll5_count_pass_epa AS roll5_avg_pass_epa,
        CASE WHEN roll5_count_rush_plays > 0 THEN CAST(roll5_sum_rush_epa AS REAL) / roll5_count_rush_plays ELSE 0.0 END AS roll5_avg_rush_epa,
        CASE WHEN roll5_count_success > 0 THEN CAST(roll5_sum_success AS REAL) / roll5_count_success ELSE 0.0 END AS roll5_success_rate,
        CASE WHEN roll5_count_cp > 0 THEN CAST(roll5_sum_cp AS REAL) / roll5_count_cp ELSE 0.0 END AS roll5_cp,
        CAST(roll5_sum_cpoe AS REAL) / roll5_count_cpoe AS roll5_avg_cpoe,
        CASE WHEN roll5_count_scrimmage_plays > 0 THEN CAST(roll5_sum_pass_plays AS REAL) / roll5_count_scrimmage_plays ELSE 0.0 END AS roll5_pass_rate,
        CAST(roll5_sum_pass_rate_oe AS REAL) / roll5_count_pass_rate_oe AS roll5_pass_rate_oe,
        CASE WHEN roll5_count_time_to_throw > 0 THEN CAST(roll5_sum_time_to_throw AS REAL) / roll5_count_time_to_throw ELSE 0.0 END AS roll5_time_to_throw,
        CASE WHEN roll5_count_yac > 0 THEN CAST(roll5_sum_yac AS REAL) / roll5_count_yac ELSE 0.0 END AS roll5_avg_yac,
        CASE WHEN roll5_count_air_yards > 0 THEN CAST(roll5_sum_air_yards AS REAL) / roll5_count_air_yards ELSE 0.0 END AS roll5_avg_air_yards,
        CASE WHEN roll5_count_rush_plays > 0 THEN CAST(roll5_sum_rush_yards AS REAL) / roll5_count_rush_plays ELSE 0.0 END AS roll5_yards_per_rush,
        CASE WHEN roll5_count_pass_plays > 0 THEN CAST(roll5_sum_pass_yards AS REAL) / roll5_count_pass_plays ELSE 0.0 END AS roll5_yards_per_pass,
        
        -- Volume averages over 5-game window
        CAST(roll5_count_rush_plays AS REAL) / 5.0 AS roll5_avg_rush_plays,
        CAST(roll5_count_pass_plays AS REAL) / 5.0 AS roll5_avg_pass_plays,
        
        -- Fixed rate metrics with explicit float casting
        CASE WHEN roll5_count_pressure > 0 THEN CAST(roll5_sum_pressure AS REAL) / roll5_count_pressure ELSE 0.30 END AS roll5_pressure_rate,
        CASE WHEN roll5_count_qb_hit > 0 THEN CAST(roll5_sum_qb_hit AS REAL) / roll5_count_qb_hit ELSE 0.0 END AS roll5_qb_hit_rate,
        CASE WHEN roll5_count_blitz > 0 THEN CAST(roll5_sum_blitz AS REAL) / roll5_count_blitz ELSE 0.0 END AS roll5_blitz_rate,
        CASE WHEN roll5_count_defenders_in_box > 0 THEN CAST(roll5_sum_defenders_in_box AS REAL) / roll5_count_defenders_in_box ELSE 0.0 END AS roll5_avg_defenders_in_box,
        CASE WHEN roll5_count_zone > 0 THEN CAST(roll5_sum_zone AS REAL) / roll5_count_zone ELSE 0.0 END AS roll5_zone_rate,
        CASE WHEN roll5_count_shotgun_spread > 0 THEN CAST(roll5_sum_shotgun_spread AS REAL) / roll5_count_shotgun_spread ELSE 0.0 END AS roll5_shotgun_spread_rate,
        CASE WHEN roll5_count_heavy_formation > 0 THEN CAST(roll5_sum_heavy_formation AS REAL) / roll5_count_heavy_formation ELSE 0.0 END AS roll5_heavy_formation_rate,
        
        -- Compute Opponent true weighted averages with float casting
        CAST(roll5_opp_sum_epa AS REAL) / roll5_opp_count_epa AS roll5_opp_avg_epa,
        CAST(roll5_opp_sum_pass_epa AS REAL) / roll5_opp_count_pass_epa AS roll5_opp_avg_pass_epa,
        CAST(roll5_opp_sum_rush_epa AS REAL) / roll5_opp_count_rush_epa AS roll5_opp_avg_rush_epa,
        CASE WHEN roll5_opp_count_success > 0 THEN CAST(roll5_opp_sum_success AS REAL) / roll5_opp_count_success ELSE 0.0 END AS roll5_opp_success_rate,
        CASE WHEN roll5_opp_count_rush_plays > 0 THEN CAST(roll5_opp_sum_rush_yards AS REAL) / roll5_opp_count_rush_plays ELSE 0.0 END AS roll5_opp_yards_per_rush,
        CASE WHEN roll5_opp_count_pass_plays > 0 THEN CAST(roll5_opp_sum_pass_yards AS REAL) / roll5_opp_count_pass_plays ELSE 0.0 END AS roll5_opp_yards_per_pass,
        CASE WHEN roll5_opp_count_pressure > 0 THEN CAST(roll5_opp_sum_pressure AS REAL) / roll5_opp_count_pressure ELSE 0.30 END AS roll5_opp_pressure_rate,
        
        -- Opponent volume averages
        CAST(roll5_opp_count_rush_plays AS REAL) / 5.0 AS roll5_opp_avg_rush_plays,
        CAST(roll5_opp_count_pass_plays AS REAL) / 5.0 AS roll5_opp_avg_pass_plays
        
    FROM rolling_sums;
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    print("rolling averages view created cleanly")


def calculate_shannon_entropy(lineup_history_string):
    """Parses a 5-game rolling string of play-level lineups, ensures exactly

    11 players are present, enforces unique combinations, and returns the Shannon Entropy.
    """
    if not lineup_history_string or pd.isna(lineup_history_string):
        return None

    plays = [play.strip() for play in lineup_history_string.split("|") if play.strip()]

    if not plays:
        return 0.0

    standardized_lineups = []

    for play in plays:
        # Split the play into individual player IDs
        player_ids = [pid for pid in play.split(";") if pid.strip()]

        # GUARDRAIL: Only accept plays that have exactly 11 players
        if len(player_ids) == 11:
            # Sort alphabetically to treat it as a unique combination
            player_ids.sort()
            combination_string = ";".join(player_ids)
            standardized_lineups.append(combination_string)

    # If no plays had exactly 11 players, return 0.0 variance
    if not standardized_lineups:
        return 0.0

    # Calculate frequency counts and snap proportions
    total_snaps = len(standardized_lineups)
    lineup_counts = Counter(standardized_lineups)

    entropy = 0.0
    for count in lineup_counts.values():
        p_i = count / total_snaps
        entropy -= p_i * math.log2(p_i)

    return entropy


def create_entropy():
    with get_db_connection(read_only=True) as conn:
        df = pd.read_sql(
            "SELECT game_id, is_home, roll5_team_offense_history,"
            " roll5_team_defense_history FROM nfl_data_5",
            conn,
        )

    offense = df["roll5_team_offense_history"]
    defense = df["roll5_team_defense_history"]

    offense_entropy = []
    defense_entropy = []

    # 2. Process metrics row-by-row
    for string in offense:
        offense_entropy.append(calculate_shannon_entropy(string))

    for string in defense:
        defense_entropy.append(calculate_shannon_entropy(string))

    # 3. Build out the dataframe rows
    df["roll5_offense_entropy"] = offense_entropy
    df["roll5_defense_entropy"] = defense_entropy
    df = df.drop(columns=["roll5_team_offense_history", "roll5_team_defense_history"])

    with get_db_connection() as conn:
        df.to_sql("nfl_model_dataset_staging", conn, if_exists="replace", index=False)
        # SQLite validates dependent views during table renames, so remove the
        # downstream views first and recreate the ones used by the current build.
        _drop_object(conn, "nfl_data_8")
        _drop_object(conn, "nfl_data_7")
        _drop_object(conn, "nfl_data_6")
        conn.execute("DROP TABLE IF EXISTS nfl_model_dataset")
        conn.execute(
            "ALTER TABLE nfl_model_dataset_staging RENAME TO nfl_model_dataset"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_model_dataset_game_side "
            "ON nfl_model_dataset(game_id, is_home)"
        )
        conn.execute(
            """
            CREATE VIEW nfl_data_6 AS
            SELECT
                a.*,
                b.roll5_offense_entropy,
                b.roll5_defense_entropy
            FROM nfl_data_5 AS a
            JOIN nfl_model_dataset AS b
              ON a.game_id = b.game_id
             AND a.is_home = b.is_home
            """
        )
    print("Lineup entropy features created.")


def finalize_features():
    conn = get_db_connection()
    cursor = conn.cursor()

    _drop_object(conn, FINAL_FEATURE_VIEW)
    query = """
    CREATE VIEW nfl_data_8 AS
    WITH weather_baselines AS (
        SELECT 
            team,
            week,
            AVG(temp_adjusted) AS avg_temp,
            AVG(wind_adjusted) AS avg_wind
        FROM nfl_data_6
        GROUP BY team, week
    )
    SELECT 
        -- 1. Game Details & Metadata
        h.game_id,
        h.season,
        h.week,
        h.team AS home_team,
        h.opponent AS away_team,
        h.team_score AS home_score,
        h.opponent_score AS away_score,
        h.spread_line,
        
        h.team_offense_points AS home_offense_points,
        h.team_offense_touchdowns AS home_offense_touchdowns,
        h.team_field_goals AS home_field_goals,
        h.opponent_offense_points AS away_offense_points,
        h.opponent_offense_touchdowns AS away_offense_touchdowns,
        h.opponent_field_goals AS away_field_goals,
        h.team_pass_ypp AS home_pass_ypp,
        h.team_rush_ypr AS home_rush_ypr,
        h.opponent_pass_yards_pp AS away_pass_ypp,
        h.opponent_rush_yards_pr AS away_rush_ypr,
        h.team_pass_epa_pp AS home_pass_epa,
        h.team_rush_epa_pr AS home_rush_epa,
        h.opponent_pass_epa_pp AS away_pass_epa,
        h.opponent_rush_epa_pr AS away_rush_epa,
        h.team_pressure_rate AS home_pressure_rate,
        h.opponent_pressure_rate AS away_pressure_rate,
        h.team_pass_plays AS home_pass_plays,
        h.team_rush_plays AS home_rush_plays,
        h.opponent_pass_plays AS away_pass_plays,
        h.opponent_rush_plays AS away_rush_plays,
        h.team_success_rate AS home_success_rate,
        h.opponent_success_rate AS away_success_rate,
        
        h.div_game,
        h.roof_adjusted,
        
        -- Game-Day Weather (with stadium-week fallback)
        COALESCE(h.temp_adjusted, wb.avg_temp) AS game_temp,
        COALESCE(h.wind_adjusted, wb.avg_wind) AS game_wind,


        -- Use for stage 2
        
        h.roll5_avg_offense_points AS home_avg_offense_points,
        h.roll5_avg_offense_points_allowed AS home_avg_offense_points_allowed,
        h.roll5_avg_offense_touchdowns - h.roll5_avg_offense_touchdowns_allowed AS diff_home_avg_offense_touchdowns,
        h.roll5_avg_field_goals - h.roll5_avg_field_goals_allowed AS diff_home_avg_field_goals,
        a.roll5_avg_offense_points AS away_avg_offense_points,
        a.roll5_avg_offense_points_allowed AS away_avg_offense_points_allowed,
        a.roll5_avg_offense_touchdowns - a.roll5_avg_offense_touchdowns_allowed AS diff_away_avg_offense_touchdowns,
        a.roll5_avg_field_goals - a.roll5_avg_field_goals_allowed AS diff_away_avg_field_goals,

        h.roll5_avg_offense_points - h.roll5_avg_offense_points_allowed AS home_diff_offensive_points,
        a.roll5_avg_offense_points - a.roll5_avg_offense_points_allowed AS away_diff_offensive_points,
        h.roll5_avg_points_above_spread - a.roll5_avg_points_above_spread AS diff_avg_points_above_spread,
        h.roll5_avg_points_above_spread AS home_avg_points_above_spread,
        a.roll5_avg_points_above_spread AS away_avg_points_above_spread,
        h.roll5_avg_result_plus_spread - a.roll5_avg_result_plus_spread AS diff_avg_result_plus_spread,
        h.roll5_avg_result_plus_spread AS home_avg_result_plus_spread,
        a.roll5_avg_result_plus_spread AS away_avg_result_plus_spread,


        -- Use for stage 1 (
        
        h.roll5_avg_pass_epa AS home_avg_pass_epa,
        h.roll5_opp_avg_pass_epa AS home_opp_avg_pass_epa,
        
        a.roll5_avg_pass_epa AS away_avg_pass_epa,
        a.roll5_opp_avg_pass_epa AS away_opp_avg_pass_epa,
        
        h.roll5_avg_rush_epa AS home_avg_rush_epa,
        h.roll5_opp_avg_rush_epa AS home_opp_avg_rush_epa,
        
        a.roll5_avg_rush_epa AS away_avg_rush_epa,
        a.roll5_opp_avg_rush_epa AS away_opp_avg_rush_epa,
        
        h.roll5_yards_per_rush AS home_avg_yards_pr,
        a.roll5_opp_yards_per_rush AS away_opp_avg_yards_pr,
        
        h.roll5_opp_yards_per_rush AS home_opp_avg_yards_pr,
        a.roll5_yards_per_rush AS away_avg_yards_pr,

        
        h.roll5_yards_per_pass AS home_avg_yards_pp,
        a.roll5_opp_yards_per_pass AS away_opp_avg_yards_pp,
        
        h.roll5_opp_yards_per_pass AS home_opp_avg_yards_pp,
        a.roll5_yards_per_pass AS away_avg_yards_pp,

        
        h.roll5_success_rate AS home_avg_success_rate,
        a.roll5_opp_success_rate AS away_opp_avg_success_rate,
        
        h.roll5_opp_success_rate AS home_opp_avg_success_rate,
        a.roll5_success_rate AS away_avg_success_rate,
        

        
        h.roll5_pressure_rate AS home_avg_pressure_rate,
        a.roll5_opp_pressure_rate AS away_opp_avg_pressure_rate,
        
        h.roll5_opp_pressure_rate AS home_opp_avg_pressure_rate,
        a.roll5_pressure_rate AS away_avg_pressure_rate,


        h.roll5_avg_rush_plays AS home_avg_rush_plays,
        h.roll5_opp_avg_rush_plays AS home_opp_avg_rush_plays,
        a.roll5_avg_rush_plays AS away_avg_rush_plays,
        a.roll5_opp_avg_rush_plays AS away_opp_avg_rush_plays,

        
        h.roll5_avg_pass_plays AS home_avg_pass_plays,
        h.roll5_opp_avg_pass_plays AS home_opp_avg_pass_plays,
        
        a.roll5_avg_pass_plays AS away_avg_pass_plays,
        a.roll5_opp_avg_pass_plays AS away_opp_avg_pass_plays,

        h.roll5_avg_epa AS home_avg_epa,
        h.roll5_opp_avg_epa AS home_opp_avg_epa,
        a.roll5_avg_epa AS away_avg_epa,
        a.roll5_opp_avg_epa AS away_opp_avg_epa,
       
     -- Home Team Engineered Features
    h.roll5_avg_redzone_epa AS home_avg_redzone_epa,
    h.roll5_avg_nonredzone_epa AS home_avg_nonredzone_epa,
    h.roll5_cp AS home_cp,
    h.roll5_avg_cpoe AS home_avg_cpoe,
    h.roll5_pass_rate AS home_pass_rate,
    h.roll5_pass_rate_oe AS home_pass_rate_oe,
    h.roll5_time_to_throw AS home_time_to_throw,
    h.roll5_avg_yac AS home_avg_yac,
    h.roll5_avg_air_yards AS home_avg_air_yards,
    h.roll5_qb_hit_rate AS home_qb_hit_rate,
    h.roll5_blitz_rate AS home_blitz_rate,
    h.roll5_avg_defenders_in_box AS home_avg_defenders_in_box,
    h.roll5_zone_rate AS home_zone_rate,
    h.roll5_shotgun_spread_rate AS home_shotgun_spread_rate,
    h.roll5_heavy_formation_rate AS home_heavy_formation_rate,
    h.roll5_offense_entropy AS home_offense_entropy,
    h.roll5_defense_entropy AS home_defense_entropy,

    -- Away Team Engineered Features
    a.roll5_avg_redzone_epa AS away_avg_redzone_epa,
    a.roll5_avg_nonredzone_epa AS away_avg_nonredzone_epa,
    a.roll5_cp AS away_cp,
    a.roll5_avg_cpoe AS away_avg_cpoe,
    a.roll5_pass_rate AS away_pass_rate,
    a.roll5_pass_rate_oe AS away_pass_rate_oe,
    a.roll5_time_to_throw AS away_time_to_throw,
    a.roll5_avg_yac AS away_avg_yac,
    a.roll5_avg_air_yards AS away_avg_air_yards,
    a.roll5_qb_hit_rate AS away_qb_hit_rate,
    a.roll5_blitz_rate AS away_blitz_rate,
    a.roll5_avg_defenders_in_box AS away_avg_defenders_in_box,
    a.roll5_zone_rate AS away_zone_rate,
    a.roll5_shotgun_spread_rate AS away_shotgun_spread_rate,
    a.roll5_heavy_formation_rate AS away_heavy_formation_rate,
    a.roll5_offense_entropy AS away_offense_entropy,
    a.roll5_defense_entropy AS away_defense_entropy
    
    FROM nfl_data_6 AS h
    JOIN nfl_data_6 AS a 
        ON h.game_id = a.game_id
    LEFT JOIN weather_baselines AS wb 
        ON h.team = wb.team 
        AND h.week = wb.week
    WHERE h.is_home = 1 
      AND a.is_home = 0
      AND h.roll5_game_count = 5
      AND a.roll5_game_count = 5
    """
    cursor.execute(query)
    _drop_object(conn, f"{FINAL_FEATURE_VIEW}_staging")
    cursor.execute(
        f"CREATE TABLE {FINAL_FEATURE_VIEW}_staging AS "
        f"SELECT * FROM {FINAL_FEATURE_VIEW}"
    )
    cursor.execute(f"DROP VIEW {FINAL_FEATURE_VIEW}")
    cursor.execute(
        f"ALTER TABLE {FINAL_FEATURE_VIEW}_staging " f"RENAME TO {FINAL_FEATURE_VIEW}"
    )
    cursor.execute(
        f"CREATE UNIQUE INDEX idx_{FINAL_FEATURE_VIEW}_game_id "
        f"ON {FINAL_FEATURE_VIEW}(game_id)"
    )
    conn.commit()
    conn.close()
    print("Final model features materialized.")


REQUIRED_MODEL_COLUMNS = {
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_pass_epa",
    "away_pass_epa",
    "home_rush_epa",
    "away_rush_epa",
    "home_avg_offense_points",
    "away_avg_offense_points",
}


def validate_features(db_path=None):
    """Fail fast when the final feature view is incomplete or malformed."""
    with get_db_connection(db_path, read_only=True) as conn:
        objects = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if FINAL_FEATURE_VIEW not in objects:
            raise RuntimeError(f"Missing final feature view: {FINAL_FEATURE_VIEW}")

        columns = [
            row[1] for row in conn.execute(f"PRAGMA table_info({FINAL_FEATURE_VIEW})")
        ]
        missing = sorted(REQUIRED_MODEL_COLUMNS.difference(columns))
        if missing:
            raise RuntimeError(f"Final feature view is missing columns: {missing}")
        if len(columns) != len(set(columns)):
            raise RuntimeError("Final feature view contains duplicate column names.")

        row_count, unique_games = conn.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT game_id) FROM {FINAL_FEATURE_VIEW}"
        ).fetchone()
        if row_count == 0:
            raise RuntimeError("Final feature view contains no games.")
        if row_count != unique_games:
            raise RuntimeError(
                f"Expected one row per game; found {row_count} rows and "
                f"{unique_games} unique game IDs."
            )

        bad_teams = conn.execute(
            f"""
            SELECT COUNT(*) FROM {FINAL_FEATURE_VIEW}
            WHERE home_team IS NULL
               OR away_team IS NULL
               OR home_team = away_team
            """
        ).fetchone()[0]
        if bad_teams:
            raise RuntimeError(f"Found {bad_teams} rows with invalid team metadata.")

        bad_rates = conn.execute(
            f"""
            SELECT COUNT(*) FROM {FINAL_FEATURE_VIEW}
            WHERE home_success_rate NOT BETWEEN 0.0 AND 1.0
               OR away_success_rate NOT BETWEEN 0.0 AND 1.0
               OR home_pressure_rate NOT BETWEEN 0.0 AND 1.0
               OR away_pressure_rate NOT BETWEEN 0.0 AND 1.0
            """
        ).fetchone()[0]
        if bad_rates:
            raise RuntimeError(f"Found {bad_rates} rows with invalid rate targets.")

    return {"rows": row_count, "columns": len(columns), "unique_games": unique_games}


def build_features(*, download=False, seasons=range(2015, 2026), db_path=None):
    """Build the complete model feature set through one public entry point."""
    if db_path is not None:
        raise NotImplementedError(
            "Custom database paths are not yet supported by the legacy SQL stages."
        )
    if download:
        data_collection_to_sql(seasons=seasons)

    cut_columns()
    make_game_features()
    order_games()
    make_rolling_features()
    create_entropy()
    finalize_features()
    summary = validate_features()
    print(
        f"Feature build complete: {summary['rows']} games, "
        f"{summary['columns']} columns."
    )
    return summary


def get_features(db_path=None):
    validate_features(db_path)
    with get_db_connection(db_path, read_only=True) as conn:
        df = pd.read_sql(f"SELECT * FROM {FINAL_FEATURE_VIEW}", conn)
    return df.set_index("game_id")
