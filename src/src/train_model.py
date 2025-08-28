#!/usr/bin/env python3
"""
train_model.py

Loads 'final_labeled_dataset.csv', trains a LightGBM classifier to detect
reel vs non-reel traffic, evaluates it, and saves the trained model to
'reel_detector.joblib'.

Requirements:
    pandas, lightgbm, joblib, scikit-learn

Usage:
    python train_model.py
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Optional: label encoder to handle non-numeric labels robustly
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "../data/final/augmented_robust_dataset.csv"
MODEL_OUTPATH = "../output/models/reel_detector_robust.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame):
    if 'label' not in df.columns:
        raise KeyError("Input dataset must contain a 'label' column.")

    # Fill NaNs with 0 as requested
    df = df.fillna(0)

    # Separate features and label
    X = df.drop(columns=['label','window_start'])
    y = df['label']

    # Convert all feature columns to numeric where possible, coerce others to NaN then fill with 0
    # This avoids model crashes when non-numeric columns are present.
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Try to convert labels to numeric; if not possible, use LabelEncoder
    label_encoder = None
    y_numeric = pd.to_numeric(y, errors='coerce')
    if y_numeric.isna().any():
        # Non-numeric labels present -> use LabelEncoder
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y.astype(str))
        y_final = pd.Series(y_encoded, name='label')
        print(f"[INFO] Labels encoded with LabelEncoder. Classes: {list(label_encoder.classes_)}")
    else:
        # All numeric (or numeric-like)
        # Use integers if they are integral; else keep as numeric
        if np.all(np.equal(np.mod(y_numeric, 1), 0)):
            y_final = y_numeric.astype(int)
        else:
            y_final = y_numeric
        print(f"[INFO] Labels appear numeric. Unique labels: {np.unique(y_final)}")

    return X, y_final, label_encoder


def safe_train_test_split(X, y):
    """
    Try to use stratify=y. If it fails (e.g., only one class present or
    insufficient samples per class), fallback to a plain split and warn.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as e:
        warnings.warn(
            f"Stratified split failed ({e}). Falling back to non-stratified split.",
            UserWarning,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder=None):
    # Initialize model (as requested)
    lgbm = lgb.LGBMClassifier(random_state=RANDOM_STATE)

    print("[INFO] Training LightGBM model...")
    lgbm.fit(X_train, y_train)

    print("[INFO] Making predictions on test set...")
    y_pred = lgbm.predict(X_test)

    # Prepare labels/order for reporting and confusion matrix
    if label_encoder is not None:
        class_names = list(label_encoder.classes_)
        # labels for confusion_matrix/classification_report should be integers [0..n_classes-1]
        labels_for_metrics = list(range(len(class_names)))
    else:
        unique_labels = np.unique(np.concatenate([y_test.astype(object), y_pred.astype(object)]))
        # Keep ordering consistent
        labels_for_metrics = list(unique_labels)
        class_names = [str(l) for l in labels_for_metrics]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(
        y_test, y_pred, labels=labels_for_metrics, target_names=class_names, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels_for_metrics)

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {acc:.6f}\n")
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix (rows=true, cols=predicted):")
    # Print confusion matrix with labels for clarity
    try:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print(cm_df)
    except Exception:
        print(cm)

    return lgbm


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"\nModel trained and saved successfully to {path}")


def main():
    try:
        print(f"[INFO] Loading dataset from '{DATA_PATH}' ...")
        df = load_data(DATA_PATH)
        print(f"[INFO] Dataset loaded. Shape: {df.shape}")

        X, y, label_encoder = prepare_data(df)
        print(f"[INFO] Feature matrix shape: {X.shape}; Target vector shape: {y.shape}")

        X_train, X_test, y_train, y_test = safe_train_test_split(X, y)
        print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        model = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder=label_encoder)

        save_model(model, MODEL_OUTPATH)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
        sys.exit(1)
    except KeyError as ke:
        print(f"[ERROR] {ke}")
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
