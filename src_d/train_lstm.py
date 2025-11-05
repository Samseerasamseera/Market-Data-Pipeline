import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src_d.utils import save_df
import joblib

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_features():
    fp = os.path.join(DATA_DIR, "features", "features_with_regimes.csv")
    df = pd.read_csv(fp, index_col=0, parse_dates=True)
    return df

def create_sequences(X, y, seq_len=60):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

def main():
    print("Loading data...")
    df = load_features()
    features = ['ret', 'ema5', 'ema15', 'atr14', 'vol_60', 'total_oi_change', 'put_call_ratio_oi', 'opt_skew_proxy']
    Xraw = df[features].fillna(0).values
    yraw = df['ret'].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xraw)
    joblib.dump(scaler, os.path.join(DATA_DIR, "models", "lstm_scaler.pkl"))
    seq_len = 60
    X_seq, y_seq = create_sequences(Xs, yraw, seq_len=seq_len)
    n = len(X_seq)
    train_idx = int(n*0.7)
    val_idx = int(n*0.85)
    X_train, y_train = X_seq[:train_idx], y_seq[:train_idx]
    X_val, y_val = X_seq[train_idx:val_idx], y_seq[train_idx:val_idx]
    X_test, y_test = X_seq[val_idx:], y_seq[val_idx:]
    print(f"Shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    n_features = X_train.shape[2]
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[es], verbose=2)
    mse = model.evaluate(X_test, y_test, verbose=0)
    print("Test MSE:", mse)
    # save model
    os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
    model.save(os.path.join(DATA_DIR, "models", "lstm_model.h5"))
    print("Saved LSTM model to data/models/lstm_model.h5")
    # Save test predictions for later analysis
    preds = model.predict(X_test).flatten()
    out = pd.DataFrame({"y_true": y_test, "y_pred": preds})
    save_df("models", "lstm_test_predictions.csv", out)
    print("Done.")

if __name__ == "__main__":
    main()
