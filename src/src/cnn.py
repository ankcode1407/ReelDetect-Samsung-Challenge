# train_cnn_baseline.py
"""
Baseline 1D-CNN training script for ReelDetect traffic classification.
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Paths
# -------------------------
DATA_PATH = "data/final/augmented_robust_dataset.csv"
MODEL_SAVE_PATH = "output/models/cnn_baseline_model.h5"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# -------------------------
# Load & Prepare Data
# -------------------------
print("[INFO] Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Separate features and labels
if "label" not in df.columns:
    raise ValueError("Dataset must contain a 'label' column.")

X = df.drop(columns=["label", "window_start"], errors="ignore")
y = df["label"].values

# Scale features
print("[INFO] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
print("[INFO] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=SEED
)

# -------------------------
# Reshape for CNN
# -------------------------
# CNN expects shape: (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"[INFO] Training samples: {X_train.shape}, Test samples: {X_test.shape}")

# -------------------------
# Build Model
# -------------------------
print("[INFO] Building CNN model...")
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=32, kernel_size=3, activation="relu"),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model.summary()

# -------------------------
# Compile & Train
# -------------------------
print("[INFO] Compiling and training...")
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# -------------------------
# Evaluate
# -------------------------
print("[INFO] Evaluating on test set...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# -------------------------
# Save Model
# -------------------------
model.save(MODEL_SAVE_PATH)
print(f"[INFO] Model saved to {MODEL_SAVE_PATH}")
