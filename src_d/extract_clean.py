# src/extract_clean.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from src_d.utils import ensure_dirs, make_time_index, save_df

ensure_dirs()

INSTRUMENT = "SIM_NIFTY"
START = datetime.utcnow() - pd.Timedelta(days=7)  # last 7 days of 5-min bars
END = datetime.utcnow()
FREQ = "5T"

def simulate_ohlcv(index):
    np.random.seed(42)
    n = len(index)
    drift = 0.00001
    vol = 0.0015
    returns = np.random.normal(loc=drift, scale=vol, size=n)
    price = 18000 * np.exp(np.cumsum(returns))  # base price
    openp = price
    close = price * (1 + np.random.normal(0, 0.0005, n))
    high = np.maximum(openp, close) * (1 + np.abs(np.random.normal(0, 0.0005, n)))
    low = np.minimum(openp, close) * (1 - np.abs(np.random.normal(0, 0.0005, n)))
    volv = np.random.randint(50, 500, n) * 10
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": volv}, index=index)
    return df

def simulate_futures(spot_df):
    fut = spot_df["close"] * (1 + np.random.normal(0.0002, 0.0003, len(spot_df)))
    fut_oi = np.maximum(0, np.random.normal(1e5, 2e4, len(spot_df))).astype(int)
    df = pd.DataFrame({"fut_close": fut, "fut_oi": fut_oi}, index=spot_df.index)
    return df

def simulate_option_chain(snapshot_index):
    strikes = np.arange(16000, 20000, 50)
    rows = []
    rng = np.random.RandomState(0)
    for ts in snapshot_index:
        call_oi = rng.poisson(50, size=len(strikes)) * 100
        put_oi = rng.poisson(40, size=len(strikes)) * 100
        total_call = call_oi.sum()
        total_put = put_oi.sum()
        max_call_strike = strikes[np.argmax(call_oi)]
        max_put_strike = strikes[np.argmax(put_oi)]
        rows.append({"timestamp": ts, "total_call_oi": total_call, "total_put_oi": total_put,
                     "max_call_strike": max_call_strike, "max_put_strike": max_put_strike})
    return pd.DataFrame(rows).set_index("timestamp")

def main():
    print("Simulating raw snapshots...")
    idx = make_time_index(START, END, freq=FREQ)
    spot = simulate_ohlcv(idx)
    fut = simulate_futures(spot)
    option_snap = simulate_option_chain(idx)

    save_df("raw", "spot_raw.csv", spot)
    save_df("raw", "fut_raw.csv", fut)
    save_df("raw", "option_snap_raw.csv", option_snap)

  
    df = spot.copy()
    rolling_median = df['close'].rolling(window=12, min_periods=1).median()
    diff = np.abs(df['close'] - rolling_median)
    iqr = df['close'].rolling(12, min_periods=1).quantile(0.75) - df['close'].rolling(12, min_periods=1).quantile(0.25)
    mask = (iqr == 0) | (diff < 5 * iqr)
    df_clean = df[mask].ffill().bfill()
    merged = df_clean.join(fut).join(option_snap)

    merged['fut_oi_change'] = merged['fut_oi'].diff().fillna(0)
    merged['total_opt_oi'] = merged['total_call_oi'].fillna(0) + merged['total_put_oi'].fillna(0)
    merged['total_opt_oi_change'] = merged['total_opt_oi'].diff().fillna(0)

    save_df("cleaned", "merged_5min.csv", merged)
    print("Saved cleaned merged_5min.csv to data/cleaned/")
    print("Done.")

if __name__ == "__main__":
    main()
