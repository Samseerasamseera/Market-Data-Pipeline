# 📈 Market Data Pipeline - Quickstart

## 🧩 Overview

This project is an **end-to-end market data analysis pipeline** that
automates data extraction, cleaning, feature engineering, regime
detection, backtesting, and deep learning model training.\
Outputs can be visualized in **Power BI** for interactive exploration.

---

## ⚙️ Prerequisites

1.  **Python 3.9+** (Recommended 3.10 or 3.11)

2.  Create a virtual environment or conda environment:

    ```bash
    python -m venv env
    env\Scripts\activate    # on Windows
    source env/bin/activate   # on macOS/Linux
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Run Order

Execute the scripts in the following sequence:

```bash
python src/extract_clean.py
python src/feature_engineer.py
python src/hmm_regimes.py
python src/strategy_backtest.py
python src/train_lstm.py
python src/anomaly_explain.py
```

---

## 📂 Output Files

File Description

---

`data/cleaned/merged_5min.csv` Aggregated 5-minute OHLCV dataset
`data/features/features.csv` Engineered features for modeling
`data/regimes/regimes.csv` Market regime labels per bar
`data/backtest/trades.csv` Trade ledger generated from backtest
`data/backtest/perf_summary.csv` Performance summary of strategy
`data/models/lstm_model.h5` Trained LSTM model
`data/models/xgb_model.json` Trained XGBoost model
`report/report.pdf` Analytical report (placeholder)

---

## 📊 Power BI Dashboard Setup

1.  Open **Power BI Desktop**.

2.  Import the CSVs from your `data/` folder.

3.  Recommended visuals:

    ***

    Visualization Data Source Description

    ***

    Line Chart `merged_5min.csv` Price chart with EMA
    overlays

    Heatmap `features.csv` Option OI and
    volatility
    visualization

    Regime Timeline `regimes.csv` Market regime
    color-coded timeline

    Trade Ledger / KPIs `trades.csv`, PnL and performance
    `perf_summary.csv` metrics

    ***

## 🧠 Notes

- Make sure the `data/` directory exists before running each script.
- If any CSV is empty, check whether the previous step completed
  successfully.
- LSTM training can take several minutes depending on system
  performance.

---
