# Save as /opt/trustgate/ai_engine/refit_scaler.py
# Run: python3 /opt/trustgate/ai_engine/refit_scaler.py

import json
import time
import pickle
import numpy as np
from datetime import datetime

OUTPUT_PATH  = "/tmp/network_features.json"
SCALER_PATH  = "/opt/trustgate/model/network_scaler.pkl"
N_SAMPLES    = 120   # 120 seconds of real traffic

FEATURE_NAMES = [
    "packet_count","total_bytes",
    "valve_write_count","pump_write_count",
    "valve_read_count","pump_read_count",
    "level_access_count","dosing_access_count","state_poll_count",
    "valve_write_deviation","pump_write_deviation",
    "novel_tag_count","tag_entropy",
    "burst_component_max","rare_access_score",
    "unique_tags_this_sec","unique_src_ips",
    "reset_count","permissive_count",
]

print("=" * 60)
print("  TrustGate Scaler Refit Tool")
print("  Collecting 120s of REAL Modbus traffic features")
print("=" * 60)
print()
print("Requirements:")
print("  ✓ modbus_server.py  — running")
print("  ✓ physics_sim.py    — running")  
print("  ✓ network_extractor.py — running (sniffer fixed)")
print()

# ── Collect raw (unscaled) samples ───────────────────────────────────────────
samples    = []
last_cycle = -1
skipped    = 0

while len(samples) < N_SAMPLES:
    try:
        with open(OUTPUT_PATH) as f:
            data = json.load(f)

        cycle        = data.get("cycle", -1)
        window_events= data.get("window_events", 0)

        # IMPORTANT: Read features_raw if available (unscaled)
        # Fall back to features if not (old extractor version)
        raw = data.get("features_raw", None)
        if raw is None:
            # Old extractor — features ARE the raw values (scaler broken)
            raw = data.get("features", [])

        # Skip duplicate cycles
        if cycle == last_cycle:
            time.sleep(0.5)
            continue

        last_cycle = cycle

        # Validate
        if len(raw) != 19:
            print(f"  [SKIP] Wrong feature dim: {len(raw)}")
            time.sleep(1.0)
            continue

        if window_events == 0:
            skipped += 1
            if skipped % 10 == 0:
                print(f"  [WARN] {skipped} empty windows — "
                      f"is physics_sim.py running?")
            time.sleep(1.0)
            continue

        arr = np.array(raw, dtype=np.float64)

        # Check for NaN/Inf
        if not np.all(np.isfinite(arr)):
            print(f"  [SKIP] NaN/Inf in features")
            time.sleep(1.0)
            continue

        samples.append(arr)
        pct = len(samples) / N_SAMPLES * 100

        # Progress with key feature values
        print(f"  [{len(samples):>3}/{N_SAMPLES}] ({pct:5.1f}%) | "
              f"pkts={arr[0]:>5.0f} | "
              f"bytes={arr[1]:>7.0f} | "
              f"entropy={arr[12]:>5.3f} | "
              f"pump_w={arr[3]:>4.0f} | "
              f"window={window_events}")

    except json.JSONDecodeError:
        # File being written — retry
        time.sleep(0.2)
        continue
    except FileNotFoundError:
        print(f"  [WAIT] {OUTPUT_PATH} not found yet...")
        time.sleep(2.0)
        continue
    except Exception as e:
        print(f"  [ERROR] {e}")
        time.sleep(1.0)
        continue

    time.sleep(1.0)

# ── Analyze collected data ────────────────────────────────────────────────────
X = np.array(samples)  # shape: [120, 19]

print()
print("=" * 60)
print(f"  COLLECTED {X.shape[0]} samples x {X.shape[1]} features")
print("=" * 60)
print()
print(f"  {'Idx':<4} {'Feature':<28} {'Min':>10} {'Max':>10} "
      f"{'Median':>10} {'IQR':>10} {'Zeros%':>8}")
print(f"  {'-'*76}")

for i, name in enumerate(FEATURE_NAMES):
    col      = X[:, i]
    q10, q90 = np.percentile(col, [10, 90])
    iqr      = q90 - q10
    zero_pct = np.sum(col == 0) / len(col) * 100
    flag     = ""
    if col.min() < 0:     flag = " ← NEGATIVE"
    if zero_pct > 95:     flag = " ← ALL ZEROS"
    if iqr < 1e-8:        flag = " ← ZERO VARIANCE"

    print(f"  [{i:02d}] {name:<28} {col.min():>10.3f} {col.max():>10.3f} "
          f"{np.median(col):>10.3f} {iqr:>10.3f} {zero_pct:>7.1f}%{flag}")

# ── Fit RobustScaler ──────────────────────────────────────────────────────────
print()
print("Fitting RobustScaler (quantile_range=10-90)...")

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(quantile_range=(10.0, 90.0))
scaler.fit(X)

# Verify output quality
X_scaled      = scaler.transform(X)
extreme_count = int(np.sum(np.abs(X_scaled) > 3.0))
extreme_pct   = extreme_count / X_scaled.size * 100

print()
print("Post-scaling verification:")
print(f"  Scaled min  : {X_scaled.min():>8.4f}")
print(f"  Scaled max  : {X_scaled.max():>8.4f}")
print(f"  Scaled mean : {X_scaled.mean():>8.4f}")
print(f"  Scaled std  : {X_scaled.std():>8.4f}")
print(f"  |x| > 3.0   : {extreme_count} values ({extreme_pct:.1f}%)")

if extreme_pct < 10:
    print("  ✅ Scaler quality: GOOD")
elif extreme_pct < 25:
    print("  ⚠️  Scaler quality: ACCEPTABLE (some outliers)")
else:
    print("  ❌ Scaler quality: POOR — features may be too sparse")
    print("     Consider collecting more varied traffic samples")

# ── Save ──────────────────────────────────────────────────────────────────────
import sklearn
payload = {
    "scaler":           scaler,
    "fit_date":         datetime.now().isoformat(),
    "n_samples":        X.shape[0],
    "n_features":       X.shape[1],
    "feature_names":    FEATURE_NAMES,
    "sklearn_version":  sklearn.__version__,
    "train_stats": {
        "min":    X.min(axis=0).tolist(),
        "max":    X.max(axis=0).tolist(),
        "median": np.median(X, axis=0).tolist(),
    }
}

# Save with protocol=2 for cross-version compatibility
with open(SCALER_PATH, "wb") as f:
    pickle.dump(payload, f, protocol=2)

print()
print(f"✅ Scaler saved to: {SCALER_PATH}")
print(f"   File size: {__import__('os').path.getsize(SCALER_PATH)} bytes")
print()
print("Next steps:")
print("  1. Restart network_extractor.py  (Ctrl+C then sudo python3 ...)")
print("  2. Start inference.py")
print("  3. Watch for NORMAL classification after 35 warmup steps")
print()
