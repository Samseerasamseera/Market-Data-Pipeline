import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import os

input_path = r"D:\KLYPTTTOOO\crypto\data\features\features.csv"
output_path = r"D:\KLYPTTTOOO\crypto\data\regimes.csv"
os.makedirs(r"D:\KLYPTTTOOO\crypto\data", exist_ok=True)

print(f"Loading features from {input_path} ...")
df = pd.read_csv(input_path)

if df.empty:
    raise ValueError("❌ features.csv is empty. Run feature_engineer.py first.")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if not num_cols:
    raise ValueError("❌ No numeric columns found in features.csv for HMM.")
X = df[num_cols].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training HMM for regime detection ...")
n_states = 3  
hmm = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200, random_state=42)
hmm.fit(X_scaled)

states = hmm.predict(X_scaled)
df["regime_label"] = states


if "return" in df.columns:
    avg_return = df.groupby("regime_label")["return"].mean().sort_values()
    regime_map = {}
    regime_map[avg_return.index[0]] = "DOWNTREND"
    if len(avg_return) > 2:
        regime_map[avg_return.index[1]] = "SIDEWAYS"
    regime_map[avg_return.index[-1]] = "UPTREND"
    df["regime"] = df["regime_label"].map(regime_map)
else:
    df["regime"] = df["regime_label"].map({0: "DOWNTREND", 1: "SIDEWAYS", 2: "UPTREND"})

df.to_csv(output_path, index=False)
