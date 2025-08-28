"""
diagnose_drift.py

Create an overlaid KDE plot for a chosen numeric feature (default: `burst_cnt`) comparing:
 - Non-Reel traffic from an original labelled dataset
 - Reel traffic from the original dataset
 - New False Positive traffic (separate file)

Usage example:
    python diagnose_drift.py \
        --original ../data/final_labeled_dataset.csv \
        --new false_positive_features.csv \
        --feature burst_cnt \
        --out drift_burst_cnt.png

Requirements:
    pip install pandas numpy matplotlib seaborn scipy

Design notes / robustness:
 - Accepts different label column names; tries common defaults if not provided.
 - Handles small-sample cases: falls back to histogram if KDE can't be computed.
 - Prints basic descriptive stats (count, mean, median, std) for each group.
 - Computes Jensen-Shannon divergence between each pair of distributions
   (gives a numeric measure of distributional drift).
 - Saves the plot to the provided output path.

"""

from __future__ import annotations
import argparse
import sys
import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon

# ------------------------- Utilities -------------------------

COMMON_LABEL_COLS = ["label", "target", "class", "y", "label_name"]


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in COMMON_LABEL_COLS:
        if c in df.columns:
            return c
    # try heuristics: any column with only two unique values
    for c in df.columns:
        if df[c].nunique(dropna=True) == 2:
            return c
    return None


def safe_series(df: pd.DataFrame, feature: str) -> pd.Series:
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in dataframe. Available columns: {list(df.columns)}")
    s = pd.to_numeric(df[feature], errors="coerce").dropna()
    return s


def describe_and_print(name: str, s: pd.Series) -> None:
    print(f"\n{name} -- n={len(s)}")
    if len(s) == 0:
        print("  (no samples)")
        return
    print(f"  mean={s.mean():.4f}, median={s.median():.4f}, std={s.std():.4f}, min={s.min():.4f}, max={s.max():.4f}")


def compute_jsd(a: np.ndarray, b: np.ndarray, bins: int = 200) -> float:
    """Compute Jensen-Shannon distance between two 1D arrays using histogram approximation.

    Returns Jensen-Shannon distance (sqrt(JS divergence)), in [0, 1].
    """
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    amin, amax = np.nanmin(a), np.nanmax(a)
    bmin, bmax = np.nanmin(b), np.nanmax(b)
    lo = min(amin, bmin)
    hi = max(amax, bmax)
    if lo == hi:
        return 0.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    # add tiny constant to avoid zeros
    hist_a = hist_a + 1e-12
    hist_b = hist_b + 1e-12
    # normalize to PMFs
    pa = hist_a / np.sum(hist_a)
    pb = hist_b / np.sum(hist_b)
    return float(jensenshannon(pa, pb, base=2.0))


# ------------------------- Main plotting routine -------------------------


def plot_kdes(
    non_reel: pd.Series,
    reel: pd.Series,
    false_pos: pd.Series,
    feature: str,
    out_path: str,
    kde_bw: Optional[str] = None,
    fill_alpha: float = 0.0,
):
    """Create a single figure with three overlaid KDE (or histogram fallback) curves.

    kde_bw: passed to seaborn.kdeplot as `bw_method` (e.g. 'scott', 'silverman', or float).
    """
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # decide whether KDE is feasible (needs few distinct samples > 1)
    def can_kde(s: pd.Series) -> bool:
        return s.dropna().shape[0] > 2 and s.nunique() > 1

    # Plot functions use consistent x-limits
    combined = pd.concat([non_reel, reel, false_pos]).dropna()
    if combined.empty:
        raise ValueError("No data available across the three series to plot.")
    xmin, xmax = combined.min(), combined.max()
    xrange_padding = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
    xmin -= xrange_padding
    xmax += xrange_padding

    plotted_any = False

    if can_kde(non_reel):
        sns.kdeplot(non_reel, bw_method=kde_bw, label=f"Non-Reel (n={len(non_reel)})", lw=2, linestyle='-', fill=False, clip=(xmin, xmax), color='blue')
        plotted_any = True
    else:
        sns.histplot(non_reel, stat='density', bins=30, label=f"Non-Reel (n={len(non_reel)})", element='step', fill=False, color='blue')

    if can_kde(reel):
        sns.kdeplot(reel, bw_method=kde_bw, label=f"Reel (n={len(reel)})", lw=2, linestyle='--', fill=False, clip=(xmin, xmax), color='red')
        plotted_any = True
    else:
        sns.histplot(reel, stat='density', bins=30, label=f"Reel (n={len(reel)})", element='step', fill=False, color='red')

    if can_kde(false_pos):
        sns.kdeplot(false_pos, bw_method=kde_bw, label=f"False Positive (n={len(false_pos)})", lw=2, linestyle='-.', fill=False, clip=(xmin, xmax), color='green')
        plotted_any = True
    else:
        sns.histplot(false_pos, stat='density', bins=30, label=f"False Positive (n={len(false_pos)})", element='step', fill=False, color='green')

    plt.xlim(xmin, xmax)
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.title(f'Distribution comparison for feature: {feature}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved KDE plot to: {out_path}")


# ------------------------- CLI -------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description="Diagnose feature drift by overlaying KDEs for a feature")
    p.add_argument('--original', required=True, help='Path to final_labeled_dataset.csv (original labelled data)')
    p.add_argument('--new', required=True, help='Path to false_positive_features.csv (new problematic samples)')
    p.add_argument('--feature', default='burst_cnt', help='Feature/column name to compare (default: burst_cnt)')
    p.add_argument('--label-col', default=None, help='Label column name in original dataset (if omitted script will attempt to find one)')
    p.add_argument('--reel-label', default=None, help='Value that indicates "Reel" in the label column (default: inferred)')
    p.add_argument('--out', default='diagnose_drift_plot.png', help='Output PNG path')
    p.add_argument('--bw', default=None, help="KDE bandwidth method for seaborn (e.g. 'scott', 'silverman' or a float)")
    args = p.parse_args(argv)

    # -- load files
    if not os.path.exists(args.original):
        print(f"ERROR: original file not found: {args.original}")
        sys.exit(2)
    if not os.path.exists(args.new):
        print(f"ERROR: new file not found: {args.new}")
        sys.exit(2)

    original = pd.read_csv(args.original)
    new_fp = pd.read_csv(args.new)

    # -- detect label column
    label_col = args.label_col or find_label_column(original)
    if label_col is None:
        print("ERROR: Could not find a label column in the original dataset. Provide --label-col explicitly.")
        print(f"Columns: {list(original.columns)}")
        sys.exit(2)

    # -- infer reel label if not provided
    if args.reel_label is not None:
        reel_label_val = args.reel_label
        # try to interpret numeric
        try:
            reel_label_val = float(reel_label_val)
            if reel_label_val.is_integer():
                reel_label_val = int(reel_label_val)
        except Exception:
            pass
    else:
        # if label column has numeric 0/1, pick max as reel
        vals = pd.Series(original[label_col].dropna().unique())
        if vals.dtype.kind in 'biufc' and set(vals.astype(int).tolist()) <= {0, 1}:
            reel_label_val = 1
        else:
            # heuristic: if any label equals 'reel' or 'Reel' etc
            lower_vals = [str(v).lower() for v in vals]
            if 'reel' in lower_vals:
                reel_label_val = vals[lower_vals.index('reel')]
            else:
                # fallback: choose the value with smaller count as 'reel' (assume reel minority)
                counts = original[label_col].value_counts()
                reel_label_val = counts.idxmin()

    print(f"Using label column: {label_col}, treating reel value as: {reel_label_val}")

    # -- get series
    try:
        reel_series = safe_series(original[original[label_col] == reel_label_val], args.feature)
        non_reel_series = safe_series(original[original[label_col] != reel_label_val], args.feature)
    except KeyError as ke:
        print(str(ke))
        sys.exit(2)

    # The new dataset is assumed to be all false positives (no label needed)
    try:
        false_pos_series = safe_series(new_fp, args.feature)
    except KeyError as ke:
        print(str(ke))
        sys.exit(2)

    # print summaries
    describe_and_print('Non-Reel (original)', non_reel_series)
    describe_and_print('Reel (original)', reel_series)
    describe_and_print('False Positive (new)', false_pos_series)

    # compute JSD between each pair (non-reel vs false_pos, reel vs false_pos, reel vs non-reel)
    a = non_reel_series.to_numpy()
    b = reel_series.to_numpy()
    c = false_pos_series.to_numpy()
    print('\nJensen-Shannon distances:')
    print(f"  Non-Reel  <-> Reel        : {compute_jsd(a, b):.4f}")
    print(f"  Non-Reel  <-> False-Pos   : {compute_jsd(a, c):.4f}")
    print(f"  Reel      <-> False-Pos   : {compute_jsd(b, c):.4f}")

    # create plot
    plot_kdes(non_reel_series, reel_series, false_pos_series, args.feature, args.out, kde_bw=args.bw)


if __name__ == '__main__':
    main()
