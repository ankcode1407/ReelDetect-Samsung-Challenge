#!/usr/bin/env python3
"""
benchmark_speed.py

Measure average inference latency of a pre-trained ML/DL model.
Supports both scikit-learn/LightGBM (.joblib) and Keras (.h5/.keras) models.
"""

import argparse
import time
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import sys


def load_model(model_path):
    """Load model dynamically based on file extension."""
    if model_path.endswith(".joblib"):
        model = joblib.load(model_path)
        model_type = "joblib"
    elif model_path.endswith(".h5") or model_path.endswith(".keras"):
        model = tf.keras.models.load_model(model_path)
        model_type = "keras"
    else:
        raise ValueError(f"Unsupported model format: {model_path}")
    return model, model_type


def prepare_sample(data_path, model_type):
    """Prepare a single row from dataset for benchmarking."""
    df = pd.read_csv(data_path)

    # Drop non-feature columns if present
    drop_cols = [c for c in ["label", "window_start"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Use the first row as sample
    sample = X.iloc[0:1]

    if model_type == "keras":
        # Convert to NumPy and reshape to (1, num_features, 1)
        sample = np.expand_dims(sample.values, axis=-1)
    # For joblib/scikit models, keep as DataFrame
    return sample


def benchmark(model, model_type, sample, n_iter=1000):
    """Run inference multiple times and compute average latency."""
    print(f"[INFO] Performing warm-up inference...")
    if model_type == "keras":
        _ = model(sample, training=False)  # eager forward pass
    else:
        _ = model.predict(sample)

    times = []

    print(f"[INFO] Running {n_iter} iterations...")
    for _ in range(n_iter):
        start = time.perf_counter()
        if model_type == "keras":
            _ = model(sample, training=False)
        else:
            _ = model.predict(sample)
        end = time.perf_counter()
        times.append(end - start)

    avg_time_ms = (sum(times) / len(times)) * 1000
    return avg_time_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference speed.")
    parser.add_argument("--model_path", required=True, help="Path to trained model (.joblib or .h5/.keras)")
    parser.add_argument("--data_path", required=True, help="Path to CSV dataset")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of inference iterations")
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    n_iter = args.iterations

    print(f"[INFO] Benchmarking model: {model_path}")
    print(f"[INFO] Using dataset: {data_path}")

    try:
        model, model_type = load_model(model_path)
        sample = prepare_sample(data_path, model_type)
        avg_latency = benchmark(model, model_type, sample, n_iter)

        print(f"[INFO] Performed {n_iter} iterations.")
        print(f"[SUCCESS] Average inference time: {avg_latency:.4f} ms")

    except KeyboardInterrupt:
        print("\n[INFO] Benchmark interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
