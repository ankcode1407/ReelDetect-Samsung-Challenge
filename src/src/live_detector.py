#!/usr/bin/env python3
"""
live_detector.py

Real-time network traffic classification using a pre-trained LightGBM model.

Usage:
    python live_detector.py --interface <interface_name>

Requirements:
    pyshark, joblib, pandas, numpy

What it does:
    - Captures live packets on the given interface in 5-second windows.
    - For each window, computes the statistical features (from client's perspective)
      that match the training feature set.
    - Formats the features as a single-row pandas DataFrame and runs model.predict().
    - Prints a one-line summary per window with timestamp, packet count and prediction.
    - Cleanly stops on Ctrl+C.

Notes:
    - This script infers "client" IP in each window as the IP which sent the fewest bytes
      in that window (simple heuristic that matches offline preprocessing).
    - Some packets may not have IP layers (ARP, etc.) and are skipped.
    - Running live capture usually requires elevated privileges (root / administrator).
"""

import sys
import time
import collections
import pyshark
import joblib
import pandas as pd
import numpy as np

# Feature names must match training data exactly and preserve order
FEATURE_NAMES = [
    'down_up_byte_ratio',
    'downlink_throughput_bps',
    'psz_mean_down',
    'psz_std_down',
    'psz_p90_down',
    'iat_mean_down',
    'iat_cov_down',
    'burst_cnt',
    'burst_bytes_avg',
    'up_tiny_pkt_rate'
]

WINDOW_SECONDS = 5.0
BURST_IAT_THRESHOLD = 0.01  # 10 ms


def extract_packet_info(pkt):
    """
    Given a pyshark packet, extract (timestamp, src_ip, dst_ip, size) if possible.
    Returns a tuple or None if packet lacks necessary fields.
    """
    try:
        ts = float(pkt.sniff_timestamp)
    except Exception:
        return None

    # size: try common attributes
    size = None
    for attr in ("length", "frame_info.len", "frame_len"):
        try:
            # direct attribute access may differ across pyshark versions
            # pkt.length is commonly available
            if attr == "length" and hasattr(pkt, "length"):
                size = int(pkt.length)
                break
            if attr == "frame_info.len" and hasattr(pkt, "frame_info") and hasattr(pkt.frame_info, "len"):
                size = int(pkt.frame_info.len)
                break
            if attr == "frame_len" and hasattr(pkt, "frame_len"):
                size = int(pkt.frame_len)
                break
        except Exception:
            size = None

    if size is None:
        # fallback: try length field via get_field_by_showname (less common)
        try:
            size = int(pkt.get_field_value("frame.len") or pkt.get_field_value("frame_length") or 0)
        except Exception:
            return None

    # src/dst IP: support IPv4 and IPv6
    src = dst = None
    try:
        src = pkt.ip.src
        dst = pkt.ip.dst
    except Exception:
        try:
            src = pkt.ipv6.src
            dst = pkt.ipv6.dst
        except Exception:
            # no IP layer; skip
            return None

    return (ts, src, dst, size)


def process_window(captured_pkts, model):
    """
    Compute features for a single window of captured packets (list of raw pyshark.Packet objects).
    Returns: (features_df, parsed_packet_count, prediction)
    - features_df: single-row pandas DataFrame with columns FEATURE_NAMES
    - parsed_packet_count: number of packets successfully parsed (used for printed summary)
    - prediction: model.predict(...) result (single value) or None on error
    """
    parsed = []
    for pkt in captured_pkts:
        info = extract_packet_info(pkt)
        if info:
            parsed.append(info)

    parsed_count = len(parsed)

    # If no parsed packets, create zero-filled features and predict (graceful handling)
    if parsed_count == 0:
        zeros = {fn: 0.0 for fn in FEATURE_NAMES}
        df = pd.DataFrame([zeros], columns=FEATURE_NAMES)
        try:
            pred = model.predict(df)[0]
        except Exception:
            pred = None
        return df, parsed_count, pred

    # Determine client IP as the endpoint that sent fewer total bytes in this window
    byte_by_ip = collections.defaultdict(int)
    for ts, src, dst, size in parsed:
        byte_by_ip[src] += size

    # client = IP with minimum total bytes sent
    client_ip = min(byte_by_ip, key=byte_by_ip.get)

    # Aggregate per-direction data from client's perspective
    down_sizes = []      # sizes of packets whose dst == client_ip
    down_timestamps = [] # timestamps of downlink packets
    up_sizes = []        # sizes of packets whose src == client_ip
    up_tiny_count = 0
    down_bytes = 0
    up_bytes = 0

    for ts, src, dst, size in parsed:
        if src == client_ip:
            up_bytes += size
            up_sizes.append(size)
            if size < 200:
                up_tiny_count += 1
        elif dst == client_ip:
            down_bytes += size
            down_sizes.append(size)
            down_timestamps.append(ts)
        else:
            # Packet neither from nor to client: ignore for directional metrics
            # (these may belong to other hosts on the same interface)
            pass

    # Compute features
    # 1) down_up_byte_ratio
    down_up_byte_ratio = float(down_bytes) / max(1, up_bytes)

    # 2) downlink_throughput_bps
    downlink_throughput_bps = 8.0 * float(down_bytes) / WINDOW_SECONDS

    # 3-5) packet size stats for downlink
    if down_sizes:
        psz_mean_down = float(np.mean(down_sizes))
        # population std (ddof=0) matches many previous definitions; consistent with earlier code
        psz_std_down = float(np.std(down_sizes, ddof=0))
        psz_p90_down = float(np.percentile(down_sizes, 90))
    else:
        psz_mean_down = 0.0
        psz_std_down = 0.0
        psz_p90_down = 0.0

    # 6-7) IAT stats for downlink
    if len(down_timestamps) >= 2:
        # sort downlink packets by timestamp to compute IATs correctly
        sorted_pairs = sorted(zip(down_timestamps, down_sizes), key=lambda x: x[0])
        ts_sorted = [p[0] for p in sorted_pairs]
        sizes_sorted = [p[1] for p in sorted_pairs]
        iats = np.diff(ts_sorted)
        if iats.size > 0:
            iat_mean_down = float(np.mean(iats))
            iat_std = float(np.std(iats, ddof=0))
            iat_cov_down = float(iat_std / iat_mean_down) if iat_mean_down > 0 else 0.0
        else:
            iat_mean_down = 0.0
            iat_cov_down = 0.0
    else:
        iat_mean_down = 0.0
        iat_cov_down = 0.0
        # ensure sizes_sorted variable exists for burst logic below
        sizes_sorted = []
        ts_sorted = []

    # 8-9) Burst detection and average burst bytes
    burst_cnt = 0
    burst_bytes_avg = 0.0
    bursts = []
    if len(down_timestamps) >= 2:
        # Recompute sorted_pairs if not present
        if not ts_sorted:
            sorted_pairs = sorted(zip(down_timestamps, down_sizes), key=lambda x: x[0])
            ts_sorted = [p[0] for p in sorted_pairs]
            sizes_sorted = [p[1] for p in sorted_pairs]

        curr_count = 1
        curr_bytes = sizes_sorted[0]
        for i in range(1, len(ts_sorted)):
            iat = ts_sorted[i] - ts_sorted[i - 1]
            if iat < BURST_IAT_THRESHOLD:
                curr_count += 1
                curr_bytes += sizes_sorted[i]
            else:
                if curr_count >= 2:
                    bursts.append(curr_bytes)
                curr_count = 1
                curr_bytes = sizes_sorted[i]
        if curr_count >= 2:
            bursts.append(curr_bytes)

        burst_cnt = len(bursts)
        burst_bytes_avg = float(np.mean(bursts)) if bursts else 0.0
    else:
        burst_cnt = 0
        burst_bytes_avg = 0.0

    # 10) up_tiny_pkt_rate
    up_tiny_pkt_rate = float(up_tiny_count) / WINDOW_SECONDS

    # Build feature dict in the exact order
    feat_dict = {
        'down_up_byte_ratio': down_up_byte_ratio,
        'downlink_throughput_bps': downlink_throughput_bps,
        'psz_mean_down': psz_mean_down,
        'psz_std_down': psz_std_down,
        'psz_p90_down': psz_p90_down,
        'iat_mean_down': iat_mean_down,
        'iat_cov_down': iat_cov_down,
        'burst_cnt': burst_cnt,
        'burst_bytes_avg': burst_bytes_avg,
        'up_tiny_pkt_rate': up_tiny_pkt_rate
    }

    # Single-row DataFrame with the correct columns and order
    features_df = pd.DataFrame([feat_dict], columns=FEATURE_NAMES)

    # Model prediction
    try:
        pred = model.predict(features_df)[0]
    except Exception:
        pred = None

    return features_df, parsed_count, pred


def parse_interface_arg():
    """
    Simple command-line parsing: expect '--interface' <name>
    """
    if '--interface' in sys.argv:
        idx = sys.argv.index('--interface')
        try:
            iface = sys.argv[idx + 1]
            return iface
        except Exception:
            print("Error: --interface provided but no interface name found.")
            sys.exit(1)
    else:
        print("Usage: python live_detector.py --interface <interface_name>")
        sys.exit(1)


def main():
    interface = parse_interface_arg()

    # Load model
    model_path = "C:\\Users\\mishr\\OneDrive\\Desktop\\ReelDetect\\output\\models\\reel_detector_robust.joblib"
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"[ERROR] Model file not found: {model_path}. Train and save model first.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print(f"[INFO] Starting live capture on interface '{interface}'. Window size: {WINDOW_SECONDS}s")
    print("[INFO] Press Ctrl+C to stop.")

    # Initialize LiveCapture (persist across windows)
    try:
        capture = pyshark.LiveCapture(interface=interface)
    except Exception as e:
        print(f"[ERROR] Could not open live capture on interface '{interface}': {e}")
        sys.exit(1)

    try:
        while True:
            # sniff for WINDOW_SECONDS (this fills capture._packets)
            try:
                capture.sniff(timeout=WINDOW_SECONDS)
            except Exception as e:
                # sniff can raise if tshark not available or permissions insufficient
                print(f"[WARN] sniff error (continuing): {e}")
                time.sleep(1)
                continue

            # copy captured packets for processing and then clear the buffer
            pkts = list(getattr(capture, "_packets", []))
            try:
                # clear internal packet buffer for the next window
                if hasattr(capture, "_packets"):
                    capture._packets.clear()
            except Exception:
                # fallback: recreate capture object to reset
                try:
                    capture.close()
                except Exception:
                    pass
                capture = pyshark.LiveCapture(interface=interface)

            features_df, pkt_count, prediction = process_window(pkts, model)

            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if prediction is None:
                pred_str = "Unknown"
            else:
                # reasonable human-readable mapping for binary labels 1/0
                try:
                    p_int = int(prediction)
                    if p_int == 1:
                        pred_str = "Reel Traffic"
                    elif p_int == 0:
                        pred_str = "Non-Reel Traffic"
                    else:
                        pred_str = f"Label {prediction}"
                except Exception:
                    pred_str = f"Label {prediction}"

            print(f"[{now}] Window processed. Packets: {pkt_count} --> Prediction: {pred_str} ({prediction})")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Stopping capture and exiting...")
        try:
            capture.close()
        except Exception:
            pass
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        try:
            capture.close()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
