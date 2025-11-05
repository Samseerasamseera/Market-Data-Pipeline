import pandas as pd
import numpy as np
import os

def save_df(subdir, filename, df):
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", subdir)
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data():
    fp = os.path.join(DATA_DIR, "features", "features_with_regimes.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"❌ File not found: {fp}")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    return df


def generate_signals(df):
    df = df.copy()
    df['signal'] = 0
    mask_long = (df['ema5'].shift(1) <= df['ema15'].shift(1)) & (df['ema5'] > df['ema15'])
    mask_short = (df['ema5'].shift(1) >= df['ema15'].shift(1)) & (df['ema5'] < df['ema15'])
    df.loc[mask_long, 'signal'] = 1
    df.loc[mask_short, 'signal'] = -1
    df['entry_price'] = df['open'].shift(-1)
    return df


def backtest(df, slippage_pct=0.0001, commission_per_trade=1.0, position_size=1.0):
    trades = []
    pos = 0
    entry_price = None
    entry_idx = None
    cash = 0.0

    for idx, row in df.iterrows():
        sig = row['signal']
        if sig != 0:
            if pos != 0:
                exit_price = row['entry_price']
                if pd.isna(exit_price):
                    continue
                pnl = (exit_price - entry_price) * pos - commission_per_trade - abs(exit_price)*slippage_pct
                cash += pnl
                trades.append({
                    "entry_time": entry_idx, "exit_time": idx,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "pos": pos, "pnl": pnl
                })
                pos = 0
                entry_price = None
                entry_idx = None

            new_entry = row['entry_price']
            if np.isnan(new_entry):
                continue
            pos = sig * position_size
            entry_price = new_entry * (1 - slippage_pct * np.sign(pos))
            entry_idx = idx

    if pos != 0 and entry_price is not None:
        last_row = df.iloc[-1]
        exit_price = last_row['close']
        pnl = (exit_price - entry_price) * pos - commission_per_trade - abs(exit_price)*slippage_pct
        cash += pnl
        trades.append({
            "entry_time": entry_idx, "exit_time": df.index[-1],
            "entry_price": entry_price, "exit_price": exit_price,
            "pos": pos, "pnl": pnl
        })

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()

    perf = {
        "total_trades": len(trades_df),
        "net_pnl": trades_df['pnl'].sum() if not trades_df.empty else 0.0,
        "avg_pnl": trades_df['pnl'].mean() if not trades_df.empty else 0.0,
        "win_rate": (trades_df['pnl'] > 0).mean() if not trades_df.empty else 0.0,
        "max_drawdown": trades_df['cum_pnl'].min() if not trades_df.empty else 0.0
    }

    return trades_df, perf


def main():
    print("Loading data...")
    df = load_data()
    df = generate_signals(df)
    print("Backtesting EMA crossover strategy...")
    trades, perf = backtest(df, slippage_pct=0.0001, commission_per_trade=1.0, position_size=1.0)

    save_df("backtest", "trades.csv", trades)
    perf_df = pd.DataFrame([perf])
    save_df("backtest", "perf_summary.csv", perf_df)

    print("✅ Saved trade ledger and perf_summary.csv")
    print("Performance Summary:")
    print(perf)
    print("Done.")


if __name__ == "__main__":
    main()
