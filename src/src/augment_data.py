#!/usr/bin/env python3
"""
augment_data.py
----------------
This script augments an existing network traffic dataset by simulating
bad network conditions (latency, jitter, packet loss) to create a more
robust dataset for training a traffic classifier.
"""

import pandas as pd
import numpy as np

# ================================
# Constants
# ================================
INPUT_DATA_PATH = "../data/final/final_labeled_dataset.csv"
OUTPUT_DATA_PATH = "../data/final/augmented_robust_dataset.csv"
NUM_AUGMENTATIONS = 4

# Define column names (must exactly match the dataset structure)
COLUMN_NAMES = [
    'window_start',
    'down_up_byte_ratio',
    'downlink_throughput_bps',
    'psz_mean_down',
    'psz_std_down',
    'psz_p90_down',
    'iat_mean_down',
    'iat_cov_down',
    'burst_cnt',
    'burst_bytes_avg',
    'up_tiny_pkt_rate',
    'label'
]

# ================================
# Augmentation Function
# ================================
def augment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment a DataFrame by simulating bad network conditions.
    - Adds latency & jitter
    - Simulates packet loss
    """
    df_aug = df.copy(deep=True)

    # --- Simulate High Latency & Jitter ---
    latency_jitter_cols = ['iat_mean_down', 'iat_cov_down']
    for col in latency_jitter_cols:
        noise_factors = np.random.uniform(1.0, 1.5, size=len(df_aug))
        df_aug[col] = df_aug[col] * noise_factors

    # --- Simulate Packet Loss ---
    packet_loss_cols = ['downlink_throughput_bps', 'burst_bytes_avg']
    for col in packet_loss_cols:
        reduction_factors = np.random.uniform(0.9, 1.0, size=len(df_aug))
        df_aug[col] = df_aug[col] * reduction_factors

    return df_aug

# ================================
# Main Execution
# ================================
def main():
    print("[INFO] Loading original data...")
    df_original = pd.read_csv(
        INPUT_DATA_PATH,
        header=None,
        skiprows=1,
        names=COLUMN_NAMES
    )
    print(f"[INFO] Original dataset loaded with {len(df_original)} rows.")

    all_dfs = [df_original]

    print(f"[INFO] Generating {NUM_AUGMENTATIONS} augmented copies...")
    for i in range(NUM_AUGMENTATIONS):
        df_aug = augment_dataframe(df_original)
        all_dfs.append(df_aug)
        print(f"[INFO] Generated augmented copy #{i+1} with {len(df_aug)} rows.")

    # Combine all DataFrames
    df_final = pd.concat(all_dfs, ignore_index=True)
    print(f"[INFO] Combined dataset size: {len(df_final)} rows.")

    # Shuffle the dataset
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    print("[INFO] Shuffled the dataset.")

    # Save to CSV
    print(f"[INFO] Saving final dataset to {OUTPUT_DATA_PATH}...")
    df_final.to_csv(OUTPUT_DATA_PATH, index=False)
    print("[SUCCESS] Final robust dataset created and saved.")

if __name__ == "__main__":
    main()
