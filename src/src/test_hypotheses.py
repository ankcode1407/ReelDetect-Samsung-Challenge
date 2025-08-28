# test_hypotheses.py (Final Corrected Version)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# DATA_PATH = "final_labeled_dataset.csv"
DATA_PATH = "../data/final/augmented_robust_dataset.csv"

# --- Define the correct column names ---
COLUMN_NAMES = [
    'window_start', 'down_up_byte_ratio', 'downlink_throughput_bps',
    'psz_mean_down', 'psz_std_down', 'psz_p90_down', 'iat_mean_down',
    'iat_cov_down', 'burst_cnt', 'burst_bytes_avg', 'up_tiny_pkt_rate',
    'label'
]

# --- 1. Load the data correctly using the default comma separator ---
print(f"Loading data from {DATA_PATH}...")
# Key Change:
# REMOVED sep='\s+' and engine='python'.
# The default separator is a comma, which is correct for your file.
# We still keep the other arguments to handle the bad header row.
df = pd.read_csv(DATA_PATH, header=None, skiprows=1, names=COLUMN_NAMES)

print("--- Verifying DataFrame ---")
print(df.info())
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Value Counts for 'label' column ---")
print(df['label'].value_counts())

# Make labels human-readable for charts
df['traffic_type'] = df['label'].apply(lambda x: 'Reel' if x == 1 else 'Non-Reel')
print("\nData processing for charts...")

# --- 2. Generate Plots ---
print("Generating plots...")
plt.figure(figsize=(10, 6))
sns.violinplot(x='traffic_type', y='psz_mean_down', data=df)
plt.title('Distribution of Mean Downlink Packet Size', fontsize=16)
plt.xlabel('Traffic Type', fontsize=12)
plt.ylabel('Mean Packet Size (bytes)', fontsize=12)
# plt.savefig('hypothesis_1_violin_plot.png')
plt.savefig('../output/plots/hypothesis_1_violin_plot_augmented.png')

plt.figure(figsize=(12, 7))
sns.kdeplot(df[df['traffic_type'] == 'Reel']['psz_p90_down'], label='Reel', fill=True)
sns.kdeplot(df[df['traffic_type'] == 'Non-Reel']['psz_p90_down'], label='Non-Reel', fill=True)
plt.title('Distribution of P90 Downlink Packet Size', fontsize=16)
plt.xlabel('90th Percentile Packet Size (bytes)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
# plt.savefig('hypothesis_2_kde_plot.png')
plt.savefig('../output/plots/hypothesis_2_kde_plot_augmented.png')
print("Plots saved successfully.")

# --- 3. Get Rock-Solid Numbers ---
print("\n--- Descriptive Statistics by Traffic Type ---")
stats = df.groupby('traffic_type')[['psz_mean_down', 'psz_std_down', 'psz_p90_down']].agg(['mean', 'std', 'median'])
print(stats)

print("\nHypothesis testing complete.")