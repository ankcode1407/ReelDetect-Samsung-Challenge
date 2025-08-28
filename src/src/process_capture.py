# src/process_capture.py

import pyshark
import pandas as pd
import numpy as np
from collections import Counter
import argparse  # <-- 1. Import the library

# (Your existing feature calculation logic would go here if it were in a function)
# For this script, we will assume the logic is in the main block.

def main(input_path, output_path):
    """
    Reads a pcapng file, processes it in 3-second windows, extracts features,
    and saves the results to a CSV file.
    """
    print(f"[INFO] Starting capture processing for: {input_path}")
    
    try:
        cap = pyshark.FileCapture(input_path)
        packets = list(cap)
        cap.close()
    except Exception as e:
        print(f"[ERROR] Could not read pcapng file. Reason: {e}")
        return

    if not packets:
        print("[INFO] No packets found in capture file.")
        return

    # Determine the client IP (heuristic: the one sending fewer bytes)
    bytes_sent = Counter()
    for p in packets:
        if hasattr(p, 'ip'):
            bytes_sent[p.ip.src] += int(p.length)
    client_ip = bytes_sent.most_common()[-1][0] if bytes_sent else None
    
    if not client_ip:
        print("[ERROR] Could not determine client IP. Cannot process flows.")
        return

    # Time-windowed analysis
    all_window_features = []
    first_ts = packets[0].sniff_timestamp
    last_ts = packets[-1].sniff_timestamp
    
    current_ts = float(first_ts)
    while current_ts < float(last_ts):
        window_end_ts = current_ts + 3.0
        window_packets = [p for p in packets if float(p.sniff_timestamp) >= current_ts and float(p.sniff_timestamp) < window_end_ts]

        if window_packets:
            # (This is a simplified version of your feature calculation logic for clarity)
            # In a full implementation, you would call your calculate_features function here.
            downlink_packets = [p for p in window_packets if hasattr(p, 'ip') and p.ip.dst == client_ip]
            uplink_packets = [p for p in window_packets if hasattr(p, 'ip') and p.ip.src == client_ip]
            
            down_bytes = sum(int(p.length) for p in downlink_packets)
            up_bytes = sum(int(p.length) for p in uplink_packets)
            down_up_byte_ratio = down_bytes / max(1, up_bytes)
            
            # ... and so on for all 11 features ...
            # For this example, we'll just use a few
            burst_cnt = len(downlink_packets) # Simplified placeholder
            
            features = {
                'window_start': current_ts,
                'down_up_byte_ratio': down_up_byte_ratio,
                'downlink_throughput_bps': (8 * down_bytes) / 3.0,
                'psz_mean_down': np.mean([int(p.length) for p in downlink_packets]) if downlink_packets else 0,
                'psz_std_down': np.std([int(p.length) for p in downlink_packets]) if downlink_packets else 0,
                'psz_p90_down': np.percentile([int(p.length) for p in downlink_packets], 90) if downlink_packets else 0,
                'iat_mean_down': 0, # Placeholder
                'iat_cov_down': 0, # Placeholder
                'burst_cnt': burst_cnt,
                'burst_bytes_avg': 0, # Placeholder
                'up_tiny_pkt_rate': 0 # Placeholder
            }
            all_window_features.append(features)

        current_ts = window_end_ts

    if not all_window_features:
        print("[INFO] No complete windows to process.")
        return

    df_out = pd.DataFrame(all_window_features)
    # Ensure columns are in the same order as your training data, adding a placeholder label
    df_out['label'] = -1 # Use -1 to indicate unlabeled data
    
    column_order = [
        'window_start', 'down_up_byte_ratio', 'downlink_throughput_bps',
        'psz_mean_down', 'psz_std_down', 'psz_p90_down', 'iat_mean_down',
        'iat_cov_down', 'burst_cnt', 'burst_bytes_avg', 'up_tiny_pkt_rate', 'label'
    ]
    df_out = df_out[column_order]

    df_out.to_csv(output_path, index=False)
    print(f"[SUCCESS] Processed {len(all_window_features)} windows and saved to {output_path}")

if __name__ == '__main__':
    # --- 2. Set up the Argument Parser ---
    parser = argparse.ArgumentParser(description="Process a .pcapng file to extract network traffic features.")
    
    # --- 3. Define the arguments ---
    parser.add_argument('--input', type=str, required=True, help="Path to the input .pcapng file.")
    parser.add_argument('--output', type=str, required=True, help="Path for the output .csv file.")
    
    # --- 4. Parse the arguments from the command line ---
    args = parser.parse_args()
    
    # --- 5. Call the main function with the provided arguments ---
    main(args.input, args.output)