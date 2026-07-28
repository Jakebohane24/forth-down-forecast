import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
import sqlite3
import math
from collections import Counter
import src.processing as process
from src.training import nfl_model


if __name__ == '__main__':
    # Rebuild every feature stage with:
    # process.build_features()
    #
    # Download fresh play-by-play data and rebuild with:
    # process.build_features(download=True, seasons=range(2015, 2026))
    model = nfl_model(True)
    model.train()
    model.evaluate('val')
    #model.get_feature_importances(stage=1)
    model.get_feature_importances(stage=2)
   
