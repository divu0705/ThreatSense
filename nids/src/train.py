"""
train.py
Preprocesses NSL-KDD and trains a RandomForest classifier to predict the
attack CATEGORY (Normal / DoS / Probe / R2L / U2R) rather than the raw
41 specific attack labels — category-level prediction is both more useful
for a dashboard and more directly comparable across datasets (relevant to
your cross-dataset generalization thesis: CICIDS2017/2018 and UNSW-NB15
use different specific attack names but the same category structure).

Run:  python src/train.py
Outputs (into models/):
  - model.joblib          the trained classifier
  - encoders.joblib       LabelEncoders for protocol_type/service/flag
  - feature_columns.joblib  ordered list of feature column names
  - label_encoder.joblib  LabelEncoder for the target (category) column
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from constants import CATEGORICAL_COLUMNS, ATTACK_CATEGORY_MAP
from download_data import load_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def to_category(label: str) -> str:
    return ATTACK_CATEGORY_MAP.get(label, "Unknown")


def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    df = df.copy()
    df["category"] = df["label"].apply(to_category)

    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Unseen categories at test time (common with cross-dataset eval)
            # get mapped to a fallback value rather than crashing.
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    feature_cols = [c for c in df.columns if c not in ("label", "category")]
    X = df[feature_cols]
    y = df["category"]
    return X, y, encoders, feature_cols


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data...")
    train_df = load_dataset("NSL_KDD_Train.csv")
    test_df = load_dataset("NSL_KDD_Test.csv")

    print("Preprocessing...")
    X_train, y_train, encoders, feature_cols = preprocess(train_df, fit=True)
    X_test, y_test, _, _ = preprocess(test_df, encoders=encoders, fit=False)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    # "Unknown" category may appear only at test time (attacks not in training
    # set) — extend the label encoder so evaluation doesn't crash on them.
    unseen = set(y_test) - set(label_encoder.classes_)
    if unseen:
        label_encoder.classes_ = pd.Index(list(label_encoder.classes_) + sorted(unseen))
    y_test_enc = label_encoder.transform(y_test)

    print(f"Training RandomForest on {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=20, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train_enc)

    print("\nEvaluating on held-out NSL-KDD test set (different attack distribution)...")
    preds = model.predict(X_test)
    present_labels = sorted(set(y_test_enc) | set(preds))
    print(classification_report(
        y_test_enc, preds,
        labels=present_labels,
        target_names=label_encoder.inverse_transform(present_labels),
        zero_division=0,
    ))

    print("Saving model artifacts...")
    joblib.dump(model, os.path.join(MODELS_DIR, "model.joblib"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.joblib"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print(f"Done. Artifacts saved in {MODELS_DIR}")


if __name__ == "__main__":
    main()
