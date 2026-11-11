import os
import sys
import joblib
import numpy as np
import pandas as pd

MODULE_DIR = os.path.dirname(__file__)
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
from model_train import HybridRFXGBoost

MODEL_PATH = os.path.join(MODULE_DIR, "models", "hybrid_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODULE_DIR, "models", "preprocessor.pkl")


def artifacts_available():
    return os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)


def load_artifacts():
    if not artifacts_available():
        missing = []
        if not os.path.exists(MODEL_PATH): missing.append(MODEL_PATH)
        if not os.path.exists(PREPROCESSOR_PATH): missing.append(PREPROCESSOR_PATH)
        raise FileNotFoundError("DDoS model artifacts not found: " + ", ".join(missing))
    model = HybridRFXGBoost.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_dataframe(df: pd.DataFrame, model, preprocessor):
    feature_names = preprocessor['feature_columns']
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing {len(missing)} required CICIDS2017 feature(s). Examples: {missing[:5]}")
    X = df[feature_names].apply(pd.to_numeric, errors='coerce').values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = preprocessor['scaler'].transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    result = pd.DataFrame({
        'prediction': np.where(predictions == 1, 'DDoS Attack', 'Benign Traffic'),
        'confidence': probabilities.max(axis=1) * 100,
        'benign_probability': probabilities[:, 0] * 100,
        'ddos_probability': probabilities[:, 1] * 100,
    })
    return result
