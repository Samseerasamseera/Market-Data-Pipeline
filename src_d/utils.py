import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def ensure_dirs():
    for sub in ["raw", "cleaned", "features", "regimes", "backtest", "models"]:
        path = os.path.join(DATA_DIR, sub)
        os.makedirs(path, exist_ok=True)
    return

def make_time_index(start_ts, end_ts, freq='5T'):
    return pd.date_range(start=start_ts, end=end_ts, freq=freq)

def save_df(subfolder, name, df):
    path = os.path.join(DATA_DIR, subfolder)
    os.makedirs(path, exist_ok=True)
    fp = os.path.join(path, name)
    df.to_csv(fp, index=True)
    return fp

def load_df(fp):
    return pd.read_csv(fp, index_col=0, parse_dates=True)
