import pandas as pd
import numpy as np
from src_d.utils import load_df, save_df
import os
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_cleaned():
    fp = os.path.join(DATA_DIR, "cleaned", "merged_5min.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    return df

def compute_features(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change().fillna(0)
    df['logret'] = np.log(df['close']).diff().fillna(0)
    bars_per_day = int(6.5*60/5)  
    df['rv_30'] = df['logret'].rolling(window=30, min_periods=5).std() * np.sqrt(252*bars_per_day/len(df))  
    at = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14, fillna=True)
    df['atr14'] = at.average_true_range()
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema15'] = df['close'].ewm(span=15, adjust=False).mean()
    df['mom30'] = df['close'] / df['close'].shift(30) - 1
    df['put_call_ratio_oi'] = (df['total_put_oi'] / (df['total_call_oi'].replace({0: np.nan}))).fillna(0)
    df['total_oi'] = df['total_opt_oi']
    df['total_oi_change'] = df['total_opt_oi_change']
    df['opt_skew_proxy'] = (df['max_call_strike'].fillna(0) - df['max_put_strike'].fillna(0))
    df['vol_60'] = df['logret'].rolling(window=60, min_periods=5).std()
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

def main():
    print("Loading cleaned data...")
    df = load_cleaned()
    print("Computing features...")
    feats = compute_features(df)
    save_df("features", "features.csv", feats)
    print("Saved data/features/features.csv")
    print("Done.")

if __name__ == "__main__":
    main()
