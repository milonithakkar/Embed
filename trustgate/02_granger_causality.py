"""
TrustGate Phase 2.4 — Granger Causality Matrix
================================================
Computes pairwise Granger causality for all continuous SWaT sensors
using ONLY the A11 normal baseline (never contaminated by attack data).

Output: trustgate_data/granger_matrix.npy   — (N_cont, N_cont) float32
        trustgate_data/granger_pvalues.npy  — raw p-values

This matrix is used as a structural prior during training:
    L_granger = λ_g * ||α_sensor * (1 - C)||_2
ensuring attention weights respect causal physics.

Runtime: ~20-40 min for all 88 sensors on CPU.
Use --fast to skip pairs with near-zero variance (speeds up 5x).

python 02_granger_causality.py [--fast]
"""

import numpy as np
import pandas as pd
import pickle
import warnings
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────
A11_CSV    = r'D:\trustgate_pcaps\A11_merged.csv'
SCALER_PKL = 'trustgate_data/robust_scaler.pkl'
OUT_MAT    = 'trustgate_data/granger_matrix.npy'
OUT_PVALS  = 'trustgate_data/granger_pvalues.npy'

MAX_LAG   = 5      # seconds of lag for Granger VAR test
P_THRESH  = 0.01   # significance threshold for causal edge
N_WORKERS = 4      # parallel workers for pairwise tests
SUBSAMPLE = 10_000  # max rows to use (speeds up without losing structure)

FAST_MODE = '--fast' in sys.argv


def granger_pair(args):
    """Test if sensor X[j] Granger-causes sensor X[i]. Returns (i, j, p_min)."""
    from statsmodels.tsa.stattools import grangercausalitytests
    i, j, yi, xj, max_lag = args
    try:
        data    = np.column_stack([yi, xj])
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        p_min   = min(results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
        return i, j, p_min
    except Exception:
        return i, j, 1.0


def main():
    Path('trustgate_data').mkdir(exist_ok=True)

    # ── Load column info from scaler ──────────────────────────────
    print('[1/4] Loading A11 sensor data...')
    with open(SCALER_PKL, 'rb') as fh:
        info = pickle.load(fh)
    c_cont = info['c_cont']     # continuous sensor column names
    N_s    = len(c_cont)
    print(f'    Continuous sensors: {N_s}')

    # ── Load A11 raw (unscaled) continuous sensor values ──────────
    a11 = pd.read_csv(A11_CSV, low_memory=False)

    # Strip .Pv / .Status suffixes so column names match c_cont list
    rename = {}
    for col in a11.columns:
        if col.endswith('.Pv'):
            rename[col] = col[:-3]
        elif col.endswith('.Status'):
            rename[col] = col[:-7]
        elif col.endswith('.Speed'):
            rename[col] = col[:-6] + '_speed'
        elif col.endswith('.Alarm'):
            rename[col] = col[:-6] + '_alarm'
    a11 = a11.rename(columns=rename)

    # Coerce everything to numeric (alarm strings become NaN → 0)
    for col in a11.columns:
        if col not in ('t_stamp', 'timestamp_sgt', 'attack_name'):
            a11[col] = pd.to_numeric(a11[col], errors='coerce')

    a11 = a11.ffill(limit=5).fillna(0)
    avail = [c for c in c_cont if c in a11.columns]
    data  = a11[avail].values.astype(np.float64)

    # Subsample for speed (Granger is O(N) per pair; N=58k is fine but slow)
    if SUBSAMPLE and len(data) > SUBSAMPLE:
        step = max(1, len(data) // SUBSAMPLE)
        data = data[::step]
    N_rows = len(data)
    print(f'    Using {N_rows:,} time steps for {len(avail)} sensors')

    # ── Skip sensors with near-zero variance (FAST_MODE) ──────────
    if FAST_MODE:
        stds      = data.std(axis=0)
        active_idx = np.where(stds > 1e-4)[0]
        print(f'    FAST mode: {len(active_idx)} sensors have variance > 1e-4')
    else:
        active_idx = np.arange(len(avail))

    # ── Build all pairs to test ────────────────────────────────────
    print('[2/4] Building pairwise test list...')
    pairs = [(i, j) for i in active_idx for j in active_idx if i != j]
    print(f'    Testing {len(pairs):,} ordered pairs '
          f'({len(active_idx)} x {len(active_idx)-1})')

    # ── Run Granger tests in parallel ─────────────────────────────
    print(f'[3/4] Running Granger tests (lag={MAX_LAG}, workers={N_WORKERS})...')

    C      = np.zeros((N_s, N_s), dtype=np.float32)
    Pvals  = np.ones( (N_s, N_s), dtype=np.float32)

    args_list = [(i, j, data[:, i], data[:, j], MAX_LAG) for i, j in pairs]
    done = 0
    edges = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(granger_pair, a): a[:2] for a in args_list}
        for fut in as_completed(futures):
            i, j, p = fut.result()
            Pvals[i, j] = p
            if p < P_THRESH:
                C[i, j] = 1.0
                edges += 1
            done += 1
            if done % 500 == 0:
                pct = done / len(pairs) * 100
                print(f'    {done:>6}/{len(pairs)} ({pct:.1f}%)  '
                      f'causal edges so far: {edges}')

    # ── Report & save ──────────────────────────────────────────────
    print('\n[4/4] Results:')
    density = edges / max(1, len(active_idx) * (len(active_idx) - 1)) * 100
    print(f'    Total causal edges: {edges}  (density: {density:.1f}%)')
    print(f'    Sensor names: {avail[:5]}...')

    # Example: show top-5 most "caused" sensors
    in_degree = C.sum(axis=1)
    top5_idx  = np.argsort(in_degree)[::-1][:5]
    print(f'    Top 5 sensors by in-degree:')
    for idx in top5_idx:
        if idx < len(avail):
            print(f'      {avail[idx]:10s}  ← {int(in_degree[idx])} causes')

    np.save(OUT_MAT,   C)
    np.save(OUT_PVALS, Pvals)
    print(f'    Saved {OUT_MAT}  (shape {C.shape})')
    print(f'    Saved {OUT_PVALS}')
    print('\n[DONE] Run python 04_pretrain.py next')


if __name__ == '__main__':
    main()