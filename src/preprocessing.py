import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)

def basic_preprocess(df, fit_scaler=False):
    # Remove rows where target 'Class' has missing values
    df = df.dropna(subset=['Class'])

    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler_path = "artifacts/scaler.pkl"
    scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return X_scaled, y
