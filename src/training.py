import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, KFold
from src.processing import get_features

class nfl_model():

    def __init__(
        self,
        veg,
        oof_strategy="kfold",
        market_feature_set="both",
    ):
        
        if oof_strategy not in {"kfold", "timeseries"}:
            raise ValueError("oof_strategy must be 'kfold' or 'timeseries'")
        if market_feature_set not in {"corrected", "composite", "both"}:
            raise ValueError(
                "market_feature_set must be 'corrected', 'composite', or 'both'"
            )

        self.df = get_features().sort_values(["season", "week"])
        self.vegas = veg
        self.oof_strategy = oof_strategy
        self.market_feature_set = market_feature_set
        self.stage_1_models = {}
        self.stage_2_models = {}
        self.monte_carlo_model = None 
        self.stage_1_feature_map = {
        # --- PASS EPA ---
        'home_pass_epa': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_avg_pressure_rate", "home_cp", "home_avg_cpoe", 
            "home_pass_rate", "home_pass_rate_oe", "home_time_to_throw", "home_offense_entropy", 
            "away_defense_entropy", "home_diff_offensive_points", "away_diff_offensive_points", 
            "away_blitz_rate","away_avg_defenders_in_box","game_wind"
        ],
        'away_pass_epa': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_avg_pressure_rate", "away_cp", "away_avg_cpoe", 
            "away_pass_rate", "away_pass_rate_oe", "away_time_to_throw", "away_offense_entropy", 
            "home_defense_entropy", "away_diff_offensive_points", "home_diff_offensive_points", 
            "home_blitz_rate","home_avg_defenders_in_box","game_wind"
        ],
        
        # --- RUSH EPA ---
        'home_rush_epa': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_blitz_rate", "home_pass_rate", "home_pass_rate_oe", 
            "home_offense_entropy", "away_defense_entropy", "home_diff_offensive_points", 
            "away_diff_offensive_points", "away_avg_defenders_in_box","game_wind"
        ],
        'away_rush_epa': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_blitz_rate", "away_pass_rate", "away_pass_rate_oe", 
            "away_offense_entropy", "home_defense_entropy", "away_diff_offensive_points", 
            "home_diff_offensive_points", "home_avg_defenders_in_box","game_wind"
        ],
        
        # --- PASS YARDS PER PLAY (YPP) ---
        'home_pass_ypp': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_avg_pressure_rate", "home_cp", "home_avg_cpoe", 
            "home_pass_rate", "home_pass_rate_oe", "home_time_to_throw", "home_offense_entropy", 
            "away_defense_entropy", "home_diff_offensive_points", "away_diff_offensive_points", 
            "away_blitz_rate", "away_avg_defenders_in_box","game_wind"
        ],
        'away_pass_ypp': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_avg_pressure_rate", "away_cp", "away_avg_cpoe", 
            "away_pass_rate", "away_pass_rate_oe", "away_time_to_throw", "away_offense_entropy", 
            "home_defense_entropy", "away_diff_offensive_points", "home_diff_offensive_points", 
            "home_blitz_rate", "home_avg_defenders_in_box","game_wind"
        ],
        
        # --- RUSH YARDS PER RUSH (YPR) ---
        'home_rush_ypr': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_blitz_rate", "home_pass_rate", "home_pass_rate_oe", 
            "home_offense_entropy", "away_defense_entropy", "home_diff_offensive_points", 
            "away_diff_offensive_points", "away_avg_defenders_in_box","game_wind"
        ],
        'away_rush_ypr': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_blitz_rate", "away_pass_rate", "away_pass_rate_oe", 
            "away_offense_entropy", "home_defense_entropy", "away_diff_offensive_points", 
            "home_diff_offensive_points", "home_avg_defenders_in_box","game_wind"
        ],
        
        # --- PASS PLAYS (VOLUME) ---
        'home_pass_plays': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_avg_pressure_rate", "home_cp", "home_avg_cpoe", 
            "home_pass_rate", "home_pass_rate_oe", "home_time_to_throw", "home_offense_entropy", 
            "away_defense_entropy", "home_diff_offensive_points", "away_diff_offensive_points", 
            "away_blitz_rate", "away_avg_defenders_in_box", "home_avg_pass_plays", "home_opp_avg_pass_plays", 
            "away_avg_pass_plays", "away_opp_avg_pass_plays","game_wind"
        ],
        'away_pass_plays': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_avg_pressure_rate", "away_cp", "away_avg_cpoe", 
            "away_pass_rate", "away_pass_rate_oe", "away_time_to_throw", "away_offense_entropy", 
            "home_defense_entropy", "away_diff_offensive_points", "home_diff_offensive_points", 
            "home_blitz_rate", "home_avg_defenders_in_box", "away_avg_pass_plays", "away_opp_avg_pass_plays", 
            "home_avg_pass_plays", "home_opp_avg_pass_plays","game_wind"
        ],
        
        # --- RUSH PLAYS (VOLUME) ---
        'home_rush_plays': [
            "home_avg_pass_epa", "away_opp_avg_pass_epa", "home_avg_rush_epa", "away_opp_avg_rush_epa", 
            "home_opp_avg_pressure_rate", "away_blitz_rate", "home_pass_rate", "home_pass_rate_oe", 
            "home_offense_entropy", "away_defense_entropy", "home_diff_offensive_points", 
            "away_diff_offensive_points", "away_avg_defenders_in_box","game_wind"
        ],
        'away_rush_plays': [
            "away_avg_pass_epa", "home_opp_avg_pass_epa", "away_avg_rush_epa", "home_opp_avg_rush_epa", 
            "away_opp_avg_pressure_rate", "home_blitz_rate", "away_pass_rate", "away_pass_rate_oe", 
            "away_offense_entropy", "home_defense_entropy", "away_diff_offensive_points", 
            "home_diff_offensive_points", "home_avg_defenders_in_box","game_wind"
        ],
        
        # --- PRESSURE RATE ---
        'home_pressure_rate': [
            "home_avg_pressure_rate", "away_opp_avg_pressure_rate", "home_opp_avg_success_rate", 
            "away_avg_success_rate", "away_avg_pass_epa", "home_opp_avg_pass_epa", 
            "away_shotgun_spread_rate", "away_heavy_formation_rate", "away_time_to_throw", 
            "away_pass_rate", "away_pass_rate_oe", "home_avg_defenders_in_box","away_offense_entropy", "home_defense_entropy", 
            "home_diff_offensive_points", "away_diff_offensive_points", "home_blitz_rate","game_wind"
        ],
        'away_pressure_rate': [
            "away_avg_pressure_rate", "home_opp_avg_pressure_rate", "away_opp_avg_success_rate", 
            "home_avg_success_rate", "home_avg_pass_epa", "away_opp_avg_pass_epa", 
            "home_shotgun_spread_rate", "home_heavy_formation_rate", "home_time_to_throw", 
            "home_pass_rate", "home_pass_rate_oe", "away_avg_defenders_in_box","home_offense_entropy", "away_defense_entropy", 
            "away_diff_offensive_points", "home_diff_offensive_points", "away_blitz_rate","game_wind"
        ]
        }
        self.stage_1_targets = [
        'pass_epa', 'rush_epa', 'pass_ypp', 'rush_ypr', 
        'pass_plays', 'rush_plays', 'pressure_rate'
        ]
        self.stage_2_base_features = [
        "home_avg_offense_points",
        "home_avg_offense_points_allowed",
        "diff_home_avg_offense_touchdowns",
        "diff_home_avg_field_goals",
        "away_avg_offense_points",
        "away_avg_offense_points_allowed",
        "diff_away_avg_offense_touchdowns",
        "diff_away_avg_field_goals",
        "home_avg_epa",
        "away_avg_epa"
        ]
        corrected_market_features = [
            "diff_avg_points_above_spread",
            "home_avg_points_above_spread",
            "away_avg_points_above_spread",
        ]
        composite_market_features = [
            "diff_avg_result_plus_spread",
            "home_avg_result_plus_spread",
            "away_avg_result_plus_spread",
        ]
        if market_feature_set == "corrected":
            self.vegas_features = corrected_market_features
        elif market_feature_set == "composite":
            self.vegas_features = composite_market_features
        else:
            self.vegas_features = (
                corrected_market_features + composite_market_features
            )
    def train(self):
        
        oof_predictions = self.train_stage_1()
        self.train_stage_2(oof_predictions)


    def train_val_test_masks(self):
        
        train = self.df['season'].isin(range(2018,2024))
        val = self.df['season'] == 2024
        test = self.df['season'] == 2025
        return train, val, test
    
    def get_feature_cols(self):
        
        non_feature_cols = [
            'game_id', 'season', 'week', 'home_team', 'away_team', 
            'home_score', 'away_score', 'home_offense_points', 'away_offense_points',
            'home_offense_touchdowns', 'home_field_goals', 
            'away_offense_touchdowns', 'away_field_goals',
            'home_pass_ypp', 'home_rush_ypr', 'away_pass_ypp', 'away_rush_ypr',
            'home_pass_epa', 'home_rush_epa', 'away_pass_epa', 'away_rush_epa',
            'home_pressure_rate', 'away_pressure_rate',
            'home_pass_plays', 'home_rush_plays', 'away_pass_plays', 'away_rush_plays',
            'home_success_rate', 'away_success_rate'
        ]
        feature_cols = [col for col in self.df.columns if col not in non_feature_cols]
        return feature_cols
    
    def split_with_features(self, feature_cols, train_mask, val_mask, test_mask):
        
        X_train = self.df.loc[train_mask,feature_cols].copy()
        X_val = self.df.loc[val_mask,feature_cols].copy()
        X_test = self.df.loc[test_mask,feature_cols].copy()
        return X_train, X_val, X_test
    
    def train_stage_1(self):
        
        train_mask, val_mask, test_mask = self.train_val_test_masks()
        feature_cols = self.get_feature_cols()
        X_train, X_val, X_test = self.split_with_features(feature_cols, train_mask, val_mask, test_mask)
        if self.oof_strategy == "kfold":
            oof_splitter = KFold(n_splits=5, shuffle=True, random_state=24)
        else:
            oof_splitter = TimeSeriesSplit(n_splits=5)
            
        oof_predictions = pd.DataFrame(index = X_train.index)
        
        for metric in self.stage_1_targets:
            for prefix in ['home','away']:
                target_col = f"{prefix}_{metric}"
                Y_train_target = self.df.loc[train_mask, target_col].copy()
                selected_features = list(self.stage_1_feature_map.get(target_col, feature_cols))
                if self.vegas:
                    selected_features.extend(self.vegas_features)
                oof_series = pd.Series(index=X_train.index, dtype=float)
                best_hyperparams = self.hyperparameter_tuning_stage_1(target_col)
                for train_idx, val_idx in oof_splitter.split(X_train):
                    fold_X_train = X_train.iloc[train_idx][selected_features]
                    fold_Y_train = Y_train_target.iloc[train_idx]
                    fold_X_val = X_train.iloc[val_idx][selected_features]

                    model = xgb.XGBRegressor(**best_hyperparams, random_state = 24, n_jobs=1)
                    model.fit(fold_X_train,fold_Y_train)
                    oof_series.iloc[val_idx] = model.predict(fold_X_val)
    
                full_model = xgb.XGBRegressor(**best_hyperparams, random_state = 24, n_jobs=1)
                full_model.fit(X_train[selected_features],Y_train_target)
                self.stage_1_models[f"pred_{target_col}"] = full_model
    
                oof_predictions[f"pred_{target_col}"] = oof_series
    
        return oof_predictions


    def train_stage_2(self, oof_predictions):

        train_mask, val_mask, test_mask = self.train_val_test_masks()
        feature_cols = self.get_feature_cols()
        X_train, X_val, X_test = self.split_with_features(feature_cols, train_mask, val_mask, test_mask)

        oof_preds = oof_predictions.copy()
        oof_preds['pred_home_pass_yards'] = oof_preds['pred_home_pass_plays'] * oof_preds['pred_home_pass_ypp']
        oof_preds['pred_away_pass_yards'] = oof_preds['pred_away_pass_plays'] * oof_preds['pred_away_pass_ypp']
        oof_preds['pred_home_rush_yards'] = oof_preds['pred_home_rush_plays'] * oof_preds['pred_home_rush_ypr']
        oof_preds['pred_away_rush_yards'] = oof_preds['pred_away_rush_plays'] * oof_preds['pred_away_rush_ypr']

        base_features = self.stage_2_base_features.copy()
        if self.vegas:
            base_features.extend(self.vegas_features)
        
        base_train = self.df.loc[train_mask, base_features]
        X_train_stage_2 = pd.concat([oof_preds, base_train], axis=1)
        # TimeSeriesSplit cannot create honest OOF predictions for its initial
        # training window. Exclude those rows from stage-two fitting.
        X_train_stage_2 = X_train_stage_2.dropna(
            subset=oof_predictions.columns
        )

        stage_2_targets = {
        'home_offense_points': 'home_points_model',
        'away_offense_points': 'away_points_model'
        }

        for target in stage_2_targets.keys():
            Y_train_stage_2 = self.df.loc[X_train_stage_2.index, target]
            param_distributions = {
            "n_estimators": [40, 60, 80, 100, 120],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 1.0, 5.0],
            "reg_lambda": [1.0, 5.0, 10.0],
            }
            tscv = TimeSeriesSplit(n_splits=3)
            search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=24, eval_metric="rmse", n_jobs=1),
            param_distributions=param_distributions,
            n_iter=20,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=24,
            n_jobs=1
            )
            search.fit(X_train_stage_2, Y_train_stage_2)
            
            best_params = search.best_params_ 
            model = xgb.XGBRegressor(**best_params,
            #objective='count:poisson', 
            random_state=24,
            n_jobs=1
        )
            model.fit(X_train_stage_2, Y_train_stage_2)
            self.stage_2_models[stage_2_targets[target]] = model

    def predict(self, val_or_test):
        
        train_mask, val_mask, test_mask = self.train_val_test_masks()
        feature_cols = self.get_feature_cols()
        X_train, X_val, X_test = self.split_with_features(feature_cols, train_mask, val_mask, test_mask)
        if val_or_test == 'val':
            mask = val_mask
            data = X_val
        elif val_or_test == 'test':
            mask = test_mask
            data = X_test
        else:
            print("Error: parameter should be 'val' or 'test'")
            return
        
        s1_predictions = pd.DataFrame(index = data.index)
        
        for metric in self.stage_1_targets:
            for prefix in ['home','away']:
                target_col = f"{prefix}_{metric}"
                selected_features = list(self.stage_1_feature_map.get(target_col, feature_cols))
                if self.vegas:
                    selected_features.extend(self.vegas_features)
                
                s1_predictions[f"pred_{target_col}"] = self.stage_1_models[f"pred_{target_col}"].predict(data[selected_features])
                
        s1_predictions['pred_home_pass_yards'] = s1_predictions['pred_home_pass_plays'] * s1_predictions['pred_home_pass_ypp']
        s1_predictions['pred_away_pass_yards'] = s1_predictions['pred_away_pass_plays'] * s1_predictions['pred_away_pass_ypp']
        s1_predictions['pred_home_rush_yards'] = s1_predictions['pred_home_rush_plays'] * s1_predictions['pred_home_rush_ypr']
        s1_predictions['pred_away_rush_yards'] = s1_predictions['pred_away_rush_plays'] * s1_predictions['pred_away_rush_ypr']


        base_cols = list(self.stage_2_base_features)
        if self.vegas:
            base_cols.extend(self.vegas_features)
        
        base_data = self.df.loc[mask, base_cols].copy()

        data_stage_2 = pd.concat([s1_predictions, base_data], axis=1)

        data_stage_2.index = data.index
        stage_2_targets = {
        'home_offense_points': 'home_points_model',
        'away_offense_points': 'away_points_model'
        }

        s2_predictions = pd.DataFrame(index = data.index)
        
        for target in stage_2_targets.keys():
            
            s2_predictions[f"pred_{target}"] = self.stage_2_models[stage_2_targets[target]].predict(data_stage_2)
        
        return s2_predictions

            
    def monte_carlo(self, preds):
        n_games = len(preds)
        n_sims = 10000
        rng = np.random.default_rng(24)
        
        home_td_lambda = ((preds['pred_home_offense_points'].values * 0.78)/7.0)[:,None]
        away_td_lambda = ((preds['pred_away_offense_points'].values * 0.78)/7.0)[:,None]
        home_fg_lambda = ((preds['pred_home_offense_points'].values * 0.22)/3.0)[:,None]
        away_fg_lambda = ((preds['pred_away_offense_points'].values * 0.22)/3.0)[:,None]
            
        home_offense_tds = rng.poisson(lam=home_td_lambda, size=(n_games, n_sims))
        away_offense_tds = rng.poisson(lam=away_td_lambda, size=(n_games, n_sims))
        home_non_offense_tds = rng.poisson(lam=0.1428, size=(n_games, n_sims))
        away_non_offense_tds = rng.poisson(lam=0.1428, size=(n_games, n_sims))
        home_fgs = rng.poisson(lam=home_fg_lambda, size=(n_games, n_sims))
        away_fgs = rng.poisson(lam=away_fg_lambda, size=(n_games, n_sims))
            
        pred_home_scores = ((home_offense_tds + home_non_offense_tds)* 7) + (home_fgs * 3)
        pred_away_scores = ((away_offense_tds + away_non_offense_tds)* 7) + (away_fgs * 3)
        diff = pred_home_scores - pred_away_scores
        spread = np.round(np.median(diff, axis=1)*2)/2
        total = pred_home_scores + pred_away_scores
        over_under = np.round(np.mean(total, axis=1)*2)/2
        home_median = np.median(pred_home_scores, axis=1).astype(int)
        away_median = np.median(pred_away_scores, axis=1).astype(int)
        home_win_prob = np.mean(diff>0, axis=1)
        home_win = home_win_prob > 0.5
            
        results = pd.DataFrame({
            "pred_home_score": home_median,
            "pred_away_score": away_median,
            "pred_spread": spread,
            "pred_over_under": over_under,
            "home_win_prob": home_win_prob,
            "pred_home_win" : home_win
        },index=preds.index)
        return results
            


        
    def evaluate(self, val_or_test, *, verbose=True):
        train_mask, val_mask, test_mask = self.train_val_test_masks()
        if val_or_test == 'val':
            mask = val_mask
        elif val_or_test == 'test':
            mask = test_mask
        else:
            print("Error: parameter should be 'val' or 'test'")
            return

        poisson = self.predict(val_or_test)
        preds = self.monte_carlo(poisson)
        
        metadata_cols = [
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
        ]
        compare = pd.concat([self.df.loc[mask, metadata_cols], preds], axis=1)

        # 1. Straight-Up Winner Accuracy
        compare["actual_home_win"] = compare["home_score"] > compare["away_score"]
        compare["pred_win_accuracy"] = compare["pred_home_win"] == compare["actual_home_win"]
        win_accuracy = compare["pred_win_accuracy"].mean()

        actual_margin = compare["home_score"] - compare["away_score"]
        predicted_margin = compare["pred_spread"]
        # nflfastR defines spread_line as the expected home margin: positive
        # when the home team is favored and negative when the away team is.
        market_home_margin = compare["spread_line"]
        compare["home_covered"] = actual_margin > market_home_margin
        compare["model_picked_home"] = predicted_margin > market_home_margin

        # Calculate absolute point difference vs Vegas
        compare["edge"] = (predicted_margin - market_home_margin).abs()
        no_pushes = compare[actual_margin != market_home_margin]
        spread_accuracy = (
            no_pushes["model_picked_home"] == no_pushes["home_covered"]
        ).mean()
    
        # 1. All Games ATS Accuracy
        all_ats = (no_pushes["model_picked_home"] == no_pushes["home_covered"]).mean()

        # 2. High-Edge ATS Accuracy (>= 2.5 pts edge)
        high_edge = no_pushes[no_pushes["edge"] >= 1.5]
        high_edge_ats = (high_edge["model_picked_home"] == high_edge["home_covered"]).mean() if len(high_edge) > 0 else 0.0

        metrics = {
            "split": val_or_test,
            "games": len(compare),
            "home_score_mae": mean_absolute_error(
                compare["home_score"], compare["pred_home_score"]
            ),
            "away_score_mae": mean_absolute_error(
                compare["away_score"], compare["pred_away_score"]
            ),
            "margin_mae": mean_absolute_error(actual_margin, predicted_margin),
            "margin_rmse": mean_squared_error(
                actual_margin, predicted_margin
            ) ** 0.5,
            "win_accuracy": win_accuracy,
            "ats_accuracy": spread_accuracy,
            "high_edge_games": len(high_edge),
            "high_edge_ats_accuracy": high_edge_ats,
        }
        if verbose:
            print(
                f"{val_or_test}: margin MAE={metrics['margin_mae']:.3f}, "
                f"win={win_accuracy:.2%}, ATS={all_ats:.2%}, "
                f"high-edge ATS={high_edge_ats:.2%} (N={len(high_edge)})"
            )
        return metrics

    def hyperparameter_tuning_stage_1(self, target_col):
        train_mask, val_mask, test_mask = self.train_val_test_masks()
        feature_cols = self.get_feature_cols()
        X_train, X_val, X_test = self.split_with_features(feature_cols, train_mask, val_mask, test_mask)
        
        selected_features = list(self.stage_1_feature_map.get(target_col, feature_cols))
        if self.vegas:
            selected_features.extend(self.vegas_features)
            
        X_train_selected = X_train.loc[:, selected_features].copy()
        Y_train_target = self.df.loc[train_mask, target_col].copy()
        
        param_distributions = {
            "n_estimators": [40, 60, 80, 100, 120],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.1],
            "subsample": [0.6, 0.7, 0.8, 0.9],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [0, 0.1, 1.0, 5.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(random_state=24, eval_metric="rmse", n_jobs=1),
            param_distributions=param_distributions,
            n_iter=20,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=24,
            n_jobs=1
        )
        search.fit(X_train_selected, Y_train_target)
            
        return search.best_params_

    def get_feature_importances(self, stage=1):
        """Prints all feature importances for models in Stage 1 or Stage 2, sorted by importance."""
        importances = {}
        feature_cols = self.get_feature_cols()

        if stage == 1:
            for pred_name, model in self.stage_1_models.items():
                target_col = pred_name.replace("pred_", "")
                selected_features = list(self.stage_1_feature_map.get(target_col, feature_cols))
                if self.vegas:
                    selected_features.extend(self.vegas_features)
                    
                scores = pd.Series(
                    model.feature_importances_, index=selected_features
                ).sort_values(ascending=False)
                importances[pred_name] = scores
                
                print(f"\n==========================================")
                print(f"   Stage 1 Feature Importances: {pred_name}")
                print(f"==========================================")
                print(scores.to_string())

        elif stage == 2:
            s1_preds = [
                f"pred_{prefix}_{m}"
                for m in self.stage_1_targets
                for prefix in ["home", "away"]
            ]
            s1_preds += [
                "pred_home_pass_yards",
                "pred_away_pass_yards",
                "pred_home_rush_yards",
                "pred_away_rush_yards",
            ]
            base_features = list(self.stage_2_base_features)
            if self.vegas:
                base_features.extend(self.vegas_features)
                
            s2_features = s1_preds + base_features

            for model_key, model in self.stage_2_models.items():
                scores = pd.Series(
                    model.feature_importances_, index=s2_features
                ).sort_values(ascending=False)
                importances[model_key] = scores
                
                print(f"\n==========================================")
                print(f"   Stage 2 Feature Importances: {model_key}")
                print(f"==========================================")
                print(scores.to_string())

        return importances









        
