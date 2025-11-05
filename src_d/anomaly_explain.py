import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os


data_path = r"D:\KLYPTTTOOO\crypto\data\features\features_with_regimes.csv" 
pred_path = r"D:\KLYPTTTOOO\crypto\data\models\lstm_test_predictions.csv" # from train_lstm.py

df = pd.read_csv(data_path)
pred = pd.read_csv(pred_path)


def clean_float(x):
    if isinstance(x, str):
        x = x.strip("[]")   
        try:
            return float(x)
        except ValueError:
            return np.nan
    return x

for col in pred.columns:
    pred[col] = pred[col].apply(clean_float)


if 'timestamp' in df.columns and 'timestamp' in pred.columns:
    df = pd.merge(df, pred, on="timestamp", how="inner")
else:
    df = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)


if 'timestamp' not in df.columns:
    df['timestamp'] = range(len(df))  


pred_cols = [c for c in df.columns if 'pred' in c.lower()]
act_cols = [c for c in df.columns if 'actual' in c.lower() or 'close' in c.lower()]

if pred_cols and act_cols:
    df['residual'] = np.abs(df[act_cols[0]] - df[pred_cols[0]])
else:
    # fallback: take last two columns
    df['residual'] = np.abs(df.iloc[:, -2] - df.iloc[:, -1])

# Normalize
df['anomaly_score'] = (df['residual'] - df['residual'].min()) / (df['residual'].max() - df['residual'].min())


threshold = 0.8
df['is_anomaly'] = df['anomaly_score'] > threshold


plt.figure(figsize=(12, 6))

y_col = act_cols[0] if act_cols else df.columns[-2]
y_pred = pred_cols[0] if pred_cols else df.columns[-1]

plt.plot(df['timestamp'], df[y_col], label='Actual', color='blue')
plt.plot(df['timestamp'], df[y_pred], label='Predicted', color='orange')

plt.scatter(df.loc[df['is_anomaly'], 'timestamp'],
            df.loc[df['is_anomaly'], y_col],
            color='red', label='Anomaly', marker='o')

plt.title("Anomaly Detection Explanation")
plt.xlabel("Timestamp / Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()


os.makedirs("output", exist_ok=True)
df.to_csv("output/anomaly_explained.csv", index=False)
print(" Anomaly explanation completed and saved to output/anomaly_explained.csv")
