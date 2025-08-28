# dashboard.py
# ReelDetect Live Traffic Monitor (Streaming, Smooth UI, Robust)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import psutil
import threading
import queue
import time
import os
import subprocess
import shutil
from datetime import datetime
import traceback
from collections import Counter, deque
from typing import Optional, Dict, Any, List, Tuple

# ================================
# Constants & Configuration
# ================================
FEATURE_NAMES = [
    'down_up_byte_ratio', 'downlink_throughput_bps', 'psz_mean_down',
    'psz_std_down', 'psz_p90_down', 'iat_mean_down', 'iat_cov_down',
    'burst_cnt', 'burst_bytes_avg', 'up_tiny_pkt_rate'
]

WINDOW_SECONDS_DEFAULT = 5.0
WINDOW_SECONDS_MIN = 3.0
WINDOW_SECONDS_MAX = 10.0

RESULTS_QUEUE_MAX = 50      # bounded queues prevent backpressure lockups
CHART_TAIL = 100
LOG_TAIL = 200

UI_LIVE_REFRESH_SECS = 1.0  # fragment refresh cadence
SMOOTHING_ALPHA = 0.35       # EMA for probability smoothing
FLIP_HYSTERESIS = 2          # stable flips only after N consistent windows

LINEBUFFER_FLAG = "-l"       # line-buffering for streaming stdout

# ================================
# TShark detection
# ================================
def find_tshark_bin():
    found = shutil.which("tshark")
    if found:
        return found
    candidates = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

TSHARK_BIN = find_tshark_bin()

# ================================
# Streamlit cached resources
# ================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)

# ================================
# Helper Functions
# ================================
def get_interfaces() -> List[str]:
    try:
        return sorted(list(psutil.net_if_addrs().keys()))
    except Exception:
        return ["Could not find interfaces"]

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def calculate_features_from_packets(pkt_list: List[Dict[str, Any]], window_duration: float) -> Dict[str, float]:
    if not pkt_list or window_duration <= 0:
        return {feature: 0.0 for feature in FEATURE_NAMES}

    bytes_sent = Counter()
    for p in pkt_list:
        src = p.get('src') or ''
        length = p.get('length') or 0
        if src:
            try:
                bytes_sent[src] += int(length)
            except Exception:
                pass

    client_ip = bytes_sent.most_common(1)[0][0] if bytes_sent else None

    downlink_packets = [p for p in pkt_list if p.get('dst') == client_ip]
    uplink_packets = [p for p in pkt_list if p.get('src') == client_ip]

    down_bytes = sum(int(p.get('length', 0)) for p in downlink_packets)
    up_bytes = sum(int(p.get('length', 0)) for p in uplink_packets)

    down_up_byte_ratio = down_bytes / max(1, up_bytes)
    downlink_throughput_bps = (8.0 * down_bytes) / window_duration

    down_pkt_sizes = [int(p.get('length', 0)) for p in downlink_packets]
    psz_mean_down = float(np.mean(down_pkt_sizes)) if down_pkt_sizes else 0.0
    psz_std_down = float(np.std(down_pkt_sizes)) if down_pkt_sizes else 0.0
    psz_p90_down = float(np.percentile(down_pkt_sizes, 90)) if down_pkt_sizes else 0.0

    down_timestamps = sorted([p['sniff_time'] for p in downlink_packets if p.get('sniff_time') is not None])
    interarrivals = np.diff(down_timestamps) if len(down_timestamps) >= 2 else np.array([])

    iat_mean_down = float(np.mean(interarrivals)) if interarrivals.size > 0 else 0.0
    iat_std_down = float(np.std(interarrivals)) if interarrivals.size > 0 else 0.0
    iat_cov_down = float(iat_std_down / iat_mean_down) if iat_mean_down > 0 else 0.0
    if not np.isfinite(iat_cov_down):
        iat_cov_down = 0.0
    else:
        iat_cov_down = min(iat_cov_down, 10.0)

    burst_cnt = 0
    burst_bytes_avg = 0.0
    if interarrivals.size > 0 and down_pkt_sizes:
        threshold = 0.01  # 10 ms
        bursts = []
        acc = [down_pkt_sizes[0]]
        for idx, dt in enumerate(interarrivals, start=1):
            if dt > threshold:
                bursts.append(acc)
                acc = [down_pkt_sizes[idx]]
            else:
                acc.append(down_pkt_sizes[idx])
        if acc:
            bursts.append(acc)
        burst_cnt = len(bursts)
        if bursts:
            burst_bytes_avg = float(np.mean([np.sum(b) for b in bursts]))

    tiny_count = sum(1 for p in uplink_packets if int(p.get('length', 0)) < 200)
    up_tiny_pkt_rate = tiny_count / window_duration

    return {
        'down_up_byte_ratio': float(down_up_byte_ratio),
        'downlink_throughput_bps': float(downlink_throughput_bps),
        'psz_mean_down': float(psz_mean_down),
        'psz_std_down': float(psz_std_down),
        'psz_p90_down': float(psz_p90_down),
        'iat_mean_down': float(iat_mean_down),
        'iat_cov_down': float(iat_cov_down),
        'burst_cnt': int(burst_cnt),
        'burst_bytes_avg': float(burst_bytes_avg),
        'up_tiny_pkt_rate': float(up_tiny_pkt_rate)
    }

class WindowAccumulator:
    def __init__(self, win_seconds: float):
        self.win_seconds = win_seconds
        self.reset()

    def reset(self):
        self._packets: List[Dict[str, Any]] = []
        self._start_ts: Optional[float] = None

    def add_packet(self, length: int, ts: Optional[float], src: str, dst: str):
        if ts is None:
            return
        if self._start_ts is None:
            self._start_ts = ts
        self._packets.append({'length': length, 'sniff_time': ts, 'src': src, 'dst': dst})

    def maybe_rollover(self) -> Optional[Tuple[Dict[str, float], float]]:
        if self._start_ts is None:
            return None
        now_ts = self._packets[-1]['sniff_time'] if self._packets else self._start_ts
        duration = now_ts - self._start_ts
        if duration >= self.win_seconds:
            feats = calculate_features_from_packets(self._packets, max(duration, 1e-3))
            self.reset()
            return feats, duration
        return None

# ================================
# Bounded queue helpers (drop-oldest on overflow)
# ================================
def q_put_bounded(q: queue.Queue, item: Dict[str, Any]):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass

# ================================
# Capture worker (streaming via tshark stdout)
# ================================
def capture_worker(interface: str,
                   model_path: str,
                   display_filter: Optional[str],
                   stop_event: threading.Event,
                   results_q: queue.Queue,
                   health_q: queue.Queue,
                   window_seconds: float):
    try:
        model = load_model(model_path)
    except Exception:
        q_put_bounded(results_q, {'error': f"Model load failed: {traceback.format_exc()}"})
        q_put_bounded(results_q, {'status': 'Capture stopped.'})
        return

    if not TSHARK_BIN:
        q_put_bounded(results_q, {'error': "tshark binary not found."})
        q_put_bounded(results_q, {'status': 'Capture stopped.'})
        return

    cmd = [
        TSHARK_BIN, "-i", str(interface), LINEBUFFER_FLAG,
        "-T", "fields",
        "-E", "separator=|",
        "-e", "frame.len",
        "-e", "frame.time_epoch",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "ipv6.src",
        "-e", "ipv6.dst",
    ]
    if display_filter:
        cmd += ["-Y", display_filter]

    acc = WindowAccumulator(window_seconds)
    packets_seen = 0
    last_health_emit = time.time()
    prob_ema = None
    flip_buffer = deque(maxlen=FLIP_HYSTERESIS)

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    except Exception:
        q_put_bounded(results_q, {'error': f"Failed to start tshark: {traceback.format_exc()}"})
        q_put_bounded(results_q, {'status': 'Capture stopped.'})
        return

    try:
        while not stop_event.is_set():
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                if proc.poll() is not None:
                    stderr_text = ""
                    try:
                        if proc.stderr:
                            stderr_text = proc.stderr.read() or ""
                    except Exception:
                        pass
                    q_put_bounded(results_q, {'error': f"tshark exited (code {proc.returncode}). {stderr_text[:240]}".strip()})
                    break
                time.sleep(0.01)
                continue

            parts = line.strip().split("|")
            while len(parts) < 6:
                parts.append("")
            frame_len_s, ts_s, ip_src, ip_dst, ipv6_src, ipv6_dst = parts[:6]
            src = ip_src if ip_src else ipv6_src
            dst = ip_dst if ip_dst else ipv6_dst
            if not src and not dst:
                continue
            try:
                length = int(frame_len_s) if frame_len_s else 0
            except Exception:
                continue
            ts = safe_float(ts_s, None)
            if ts is None:
                continue

            acc.add_packet(length, ts, src, dst)
            packets_seen += 1

            now = time.time()
            if now - last_health_emit > 1.0:
                q_put_bounded(health_q, {'packets_seen': packets_seen, 'ts': now})
                packets_seen = 0
                last_health_emit = now

            rolled = acc.maybe_rollover()
            if rolled:
                feats, duration = rolled
                X = np.array([[feats[name] for name in FEATURE_NAMES]], dtype=np.float32)
                try:
                    proba = model.predict_proba(X)[0]
                    p1 = float(proba[1])
                except Exception:
                    try:
                        pred = int(model.predict(X)[0])
                        p1 = 1.0 if pred == 1 else 0.0
                    except Exception:
                        q_put_bounded(results_q, {'error': f"Model prediction failed: {traceback.format_exc()}"})
                        p1 = 0.0

                prob_ema = p1 if prob_ema is None else (SMOOTHING_ALPHA * p1 + (1 - SMOOTHING_ALPHA) * prob_ema)
                pred_raw = "Reel Traffic" if p1 >= 0.5 else "Non-Reel Traffic"

                flip_buffer.append(1 if p1 >= 0.5 else 0)
                if len(flip_buffer) == FLIP_HYSTERESIS and (all(v == 1 for v in flip_buffer) or all(v == 0 for v in flip_buffer)):
                    pred_label = "Reel Traffic" if flip_buffer[-1] == 1 else "Non-Reel Traffic"
                else:
                    pred_label = pred_raw

                q_put_bounded(results_q, {
                    'prediction': pred_label,
                    'prob_raw': p1,
                    'prob_ema': prob_ema,
                    'confidence': f"{(prob_ema if prob_ema is not None else p1) * 100:.2f}%",
                    'burst_cnt': int(feats.get('burst_cnt', 0)),
                    'throughput': feats.get('downlink_throughput_bps', 0.0) / 1_000_000.0,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'win_seconds': duration
                })
    except Exception:
        q_put_bounded(results_q, {'error': traceback.format_exc()})
    finally:
        try:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass
        q_put_bounded(results_q, {'status': 'Capture stopped.'})

# ================================
# Session State Initialization
# ================================
def init_session_state():
    ss = st.session_state
    if 'is_capturing' not in ss:
        ss.is_capturing = False
    if 'log_df' not in ss:
        ss.log_df = pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'burst_cnt', 'throughput'])
    if 'chart_data' not in ss:
        ss.chart_data = pd.DataFrame(columns=['burst_cnt', 'throughput'])
    if 'interfaces' not in ss:
        ss.interfaces = get_interfaces()
    if 'last_result' not in ss:
        ss.last_result = None
    if 'worker_thread' not in ss:
        ss.worker_thread = None
    if 'stop_event' not in ss:
        ss.stop_event = None
    if 'results_queue' not in ss:
        ss.results_queue = queue.Queue(maxsize=RESULTS_QUEUE_MAX)
    if 'health_queue' not in ss:
        ss.health_queue = queue.Queue(maxsize=RESULTS_QUEUE_MAX)
    if 'window_seconds' not in ss:
        ss.window_seconds = WINDOW_SECONDS_DEFAULT

# ================================
# Live fragment (smooth updates)
# ================================
def drain_results_and_update_ui(status_placeholder, confidence_placeholder, health_placeholder, chart_placeholder, log_placeholder):
    updated_any = False
    # Drain results
    try:
        while True:
            result = st.session_state.results_queue.get_nowait()
            updated_any = True
            if 'error' in result:
                st.warning(f"Worker warning/error: {result['error']}")
                st.session_state.log_df.loc[len(st.session_state.log_df)] = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'prediction': 'Error',
                    'confidence': result['error'][:12],
                    'burst_cnt': 0,
                    'throughput': 0.0
                }
                if len(st.session_state.log_df) > LOG_TAIL:
                    st.session_state.log_df = st.session_state.log_df.tail(LOG_TAIL).reset_index(drop=True)
                continue
            if 'status' in result:
                st.info(result['status'])
                st.session_state.is_capturing = False
                st.session_state.last_result = None
                break

            st.session_state.last_result = result
            st.session_state.chart_data.loc[len(st.session_state.chart_data)] = {
                'burst_cnt': result.get('burst_cnt', 0),
                'throughput': result.get('throughput', 0.0)
            }
            if len(st.session_state.chart_data) > CHART_TAIL:
                st.session_state.chart_data = st.session_state.chart_data.tail(CHART_TAIL).reset_index(drop=True)

            st.session_state.log_df.loc[len(st.session_state.log_df)] = {
                'timestamp': result.get('timestamp', datetime.now().strftime("%H:%M:%S")),
                'prediction': result.get('prediction', ''),
                'confidence': result.get('confidence', ''),
                'burst_cnt': result.get('burst_cnt', 0),
                'throughput': result.get('throughput', 0.0)
            }
            if len(st.session_state.log_df) > LOG_TAIL:
                st.session_state.log_df = st.session_state.log_df.tail(LOG_TAIL).reset_index(drop=True)
    except queue.Empty:
        pass

    # Drain health (keep last only)
    latest_health = None
    try:
        while True:
            latest_health = st.session_state.health_queue.get_nowait()
    except queue.Empty:
        pass

    # Update top metrics/labels (only touch containers)
    if st.session_state.last_result:
        last = st.session_state.last_result
        pred_color = "blue" if last.get('prediction') == "Reel Traffic" else "green"
        status_placeholder.markdown(f"### <span style='color:{pred_color};'>{last.get('prediction')}</span>", unsafe_allow_html=True)
        confidence_placeholder.metric("Model Confidence (EMA)", last.get('confidence', '—'))
    else:
        if st.session_state.is_capturing:
            status_placeholder.markdown("### Status: Capturing…")
            confidence_placeholder.metric("Model Confidence (EMA)", "…")
        else:
            status_placeholder.markdown("### Status: Stopped")
            confidence_placeholder.metric("Model Confidence (EMA)", "N/A")

    if latest_health and 'packets_seen' in latest_health:
        health_placeholder.metric("Packets/sec", f"{latest_health['packets_seen']}")
    else:
        health_placeholder.metric("Packets/sec", "—")

    # Chart & log containers
    with chart_placeholder.container():
        st.subheader("Real-time Feature Plot (last 100)")
        if not st.session_state.chart_data.empty:
            st.line_chart(st.session_state.chart_data)
        else:
            st.write("No data yet.")

    with log_placeholder.container():
        st.subheader("Prediction Log (most recent at top)")
        if not st.session_state.log_df.empty:
            st.dataframe(st.session_state.log_df.iloc[::-1], use_container_width=True)
        else:
            st.write("No logs yet.")

    return updated_any

# Use a fragment to refresh only the live section smoothly
@st.experimental_fragment(run_every=UI_LIVE_REFRESH_SECS)
def live_fragment(status_placeholder, confidence_placeholder, health_placeholder, chart_placeholder, log_placeholder):
    if st.session_state.is_capturing:
        drain_results_and_update_ui(status_placeholder, confidence_placeholder, health_placeholder, chart_placeholder, log_placeholder)
    else:
        # Still render current state without polling queues
        drain_results_and_update_ui(status_placeholder, confidence_placeholder, health_placeholder, chart_placeholder, log_placeholder)

# ================================
# Streamlit UI Application
# ================================
def init_layout():
    st.set_page_config(page_title="ReelDetect Live Monitor", layout="wide")
    st.title("🚦 ReelDetect Live Traffic Monitor")
    init_session_state()

def main():
    init_layout()

    with st.sidebar:
        st.header("▶️ Capture Controls")

        if st.button("Refresh Interfaces"):
            st.session_state.interfaces = get_interfaces()

        interface = st.selectbox("Select Network Interface", options=st.session_state.interfaces, key='interface_select')
        model_option = st.selectbox("Select Model", options=['reel_detector_robust.joblib', 'reel_detector_naive.joblib'], key='model_select')

        display_filter = st.text_input("Optional Display Filter (-Y)", value="", help="Example: not arp and not dns")

        st.markdown("**TShark**")
        st.text(f"Detected: {TSHARK_BIN or 'Not found'}")

        st.session_state.window_seconds = st.slider("Window (seconds)",
                                                    min_value=WINDOW_SECONDS_MIN,
                                                    max_value=WINDOW_SECONDS_MAX,
                                                    value=st.session_state.window_seconds,
                                                    step=0.5)

        start_btn = st.button("Start Capture", key='start_btn')
        stop_btn = st.button("Stop Capture", key='stop_btn')

    col1, col2, col3 = st.columns(3)
    with col1:
        status_placeholder = st.empty()
    with col2:
        confidence_placeholder = st.empty()
    with col3:
        health_placeholder = st.empty()

    st.markdown("---")
    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    # Start capture
    if start_btn and not st.session_state.is_capturing:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
        full_model_path = os.path.join(PROJECT_ROOT, 'output', 'models', st.session_state.model_select)

        if not os.path.exists(full_model_path):
            st.error(f"Model file not found at: {full_model_path}")
        elif not TSHARK_BIN:
            st.error("tshark binary not found. Install Wireshark/tshark and ensure it's on PATH.")
        else:
            # Reset queues and state
            st.session_state.results_queue = queue.Queue(maxsize=RESULTS_QUEUE_MAX)
            st.session_state.health_queue = queue.Queue(maxsize=RESULTS_QUEUE_MAX)
            st.session_state.log_df = pd.DataFrame(columns=['timestamp', 'prediction', 'confidence', 'burst_cnt', 'throughput'])
            st.session_state.chart_data = pd.DataFrame(columns=['burst_cnt', 'throughput'])
            stop_event = threading.Event()
            worker = threading.Thread(
                target=capture_worker,
                args=(
                    st.session_state.interface_select,
                    full_model_path,
                    (display_filter.strip() or None),
                    stop_event,
                    st.session_state.results_queue,
                    st.session_state.health_queue,
                    st.session_state.window_seconds
                ),
                daemon=True
            )
            st.session_state.update({
                'is_capturing': True,
                'worker_thread': worker,
                'stop_event': stop_event,
                'last_result': None
            })
            worker.start()
            st.success(f"Capture started on '{st.session_state.interface_select}'...")

    # Stop capture
    if stop_btn and st.session_state.is_capturing:
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        if st.session_state.worker_thread:
            st.session_state.worker_thread.join(timeout=3.0)
        st.session_state.is_capturing = False
        st.info("Stopping capture...")

    # Live, smooth updates via fragment (no explicit reruns)
    live_fragment(status_placeholder, confidence_placeholder, health_placeholder, chart_placeholder, log_placeholder)

if __name__ == '__main__':
    main()
