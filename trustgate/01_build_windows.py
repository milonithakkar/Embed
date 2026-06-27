"""
TrustGate Phase 1 — Build Windowed NPZ Dataset
================================================
Input : trustgate_data/A11_merged.csv   (NORMAL baseline — ZERO attacks)
        trustgate_data/A12_merged.csv   (attack dataset with labels)
Output: trustgate_data/swat_final.npz
        trustgate_data/robust_scaler.pkl

Run BEFORE create_labels.py if swat_final.npz does not yet exist.
If you already have swat_multilabel.npz, skip to 02_granger_causality.py
but re-run this if you need the A11 pre-training windows.

python 01_build_windows.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import pickle, warnings
warnings.filterwarnings('ignore')

# ─── PATHS — adjust to your system ───────────────────────────────
A11_CSV    = r'D:\trustgate_pcaps\A11_merged.csv'
A12_CSV    = r'D:\trustgate_pcaps\A12_merged.csv'
OUT_NPZ    = 'trustgate_data/swat_final.npz'
SCALER_PKL = 'trustgate_data/robust_scaler.pkl'

# ─── WINDOW CONFIG ────────────────────────────────────────────────
W_STD   = 30   # Standard 30-second window (main model window)
W_MICRO = 5    # Micro 5-second window  (fast DoS detection)
W_MACRO = 120  # Macro 120-second window (slow chemical attacks)
STRIDE  = 1    # 1-second stride for micro + standard
STRIDE_MACRO = 5  # 5-second stride for macro (memory saving)

# ─── SPLIT STRATEGY ─────────────────────────────────────────────
# Normal (non-attack) rows: chronological 70/15/15
# Attack events: each event split 70/15/15 internally (chronological,
# with a purge buffer) so every attack class appears in every split.
TRAIN_R = 0.70
VAL_R   = 0.15

# Attack split strategy: PER-EVENT CHRONOLOGICAL SPLIT.
# Each named attack event is split internally 70/15/15 by time (first part
# -> train, middle -> val, last -> test), with a purge buffer of PURGE_SEC
# seconds dropped around each internal cut so no window straddles a
# train/val or val/test boundary. This guarantees every attack CLASS is
# represented in every split (required for meaningful val/test metrics),
# at the cost of mild within-event similarity between adjacent splits —
# the standard practical compromise for short, few-shot attack datasets.
#
# Events that are too short to safely split (fewer than MIN_SPLIT_ROWS)
# are kept whole in TRAIN; their class still appears in val/test via
# the other (longer) events of the same class, where available.
PURGE_SEC      = W_STD   # purge buffer = one window length (30s)
MIN_SPLIT_ROWS = 4 * W_STD  # need >=4x window length to split an event safely

# ─── KNOWN SWaT SENSOR DEFINITIONS ───────────────────────────────
# Continuous: need RobustScaler. Attack values appear as z-score outliers.
CONT_SENSORS = [
    'FIT101','FIT201','FIT301','FIT401','FIT501','FIT502','FIT503','FIT504',
    'LIT101','LIT201','LIT301','LIT401','LIT501','LIT601',
    'PIT101','PIT501','PIT502','PIT503','DPIT301',
    'AIT201','AIT202','AIT203','AIT401','AIT402',
    'AIT501','AIT502','AIT503','AIT504',
]
# Binary: pump/valve states. Keep as-is (0 or 1, no scaling).
BIN_SENSORS = [
    'P101','P102','P201','P202','P203','P204','P205','P206',
    'P301','P302','P401','P402','P403','P404','P501','P502',
    'P601','P602','P603',
    'MV101','MV201','MV301','MV302','MV303','MV304','MV401',
    'MV501','MV502','MV503','MV504','UV401',
]
META_COLS   = {'label_binary','label_class','attack_name','timestamp_sgt'}
COMP_PREFIX = 'comp_'


# ─── UTILITIES ────────────────────────────────────────────────────

def normalise_sensor_cols(df):
    """
    Strip .Pv / .Status suffixes and binary-encode .Alarm columns.
    Handles CSVs where columns are still in raw SWaT historian format:
      FIT101.Pv     → FIT101
      MV101.Status  → MV101
      P501.Speed    → P501_speed
      LS201.Alarm   → LS201_alarm  (Active=1, else=0)
      P1_STATE      → P1_STATE     (kept as-is)
    If columns are already clean (no suffixes), this is a no-op.
    """
    ALARM_MAP = {'Active': 1.0, 'Inactive': 0.0, 'Bad Input': 0.0}
    rename = {}
    alarm_cols = {}

    for col in df.columns:
        if col.endswith('.Pv'):
            rename[col] = col[:-3]
        elif col.endswith('.Status'):
            rename[col] = col[:-7]
        elif col.endswith('.Speed'):
            rename[col] = col[:-6] + '_speed'
        elif col.endswith('.Alarm'):
            alarm_cols[col] = col[:-6] + '_alarm'

    # Rename straightforward suffix columns
    df = df.rename(columns=rename)

    # Encode alarm columns: string → 0/1 float
    for old, new in alarm_cols.items():
        df[new] = df[old].map(ALARM_MAP).fillna(0.0)
        df = df.drop(columns=[old])

    return df


def safe_col(df, col, default=0.0):
    """Get column or return zeros — handles missing columns gracefully.
    Also coerces to numeric in case the column came through as object/string dtype."""
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors='coerce').fillna(default)


def compute_physics_features(df):
    """
    Derive 12 physics-informed features encoding SWaT conservation laws.
    These encode domain knowledge the BiLSTM can learn from directly.
    All features should be ~constant during NORMAL operation; attacks shift them.
    """
    f = {}
    # ── Flow sensors ──────────────────────────────────────────────
    fit101 = safe_col(df,'FIT101'); fit201 = safe_col(df,'FIT201')
    fit301 = safe_col(df,'FIT301'); fit401 = safe_col(df,'FIT401')
    lit101 = safe_col(df,'LIT101'); lit301 = safe_col(df,'LIT301')
    lit401 = safe_col(df,'LIT401')
    pit501 = safe_col(df,'PIT501'); pit503 = safe_col(df,'PIT503')
    ait202 = safe_col(df,'AIT202'); ait203 = safe_col(df,'AIT203')
    p101   = safe_col(df,'P101');   mv101  = safe_col(df,'MV101')

    # 1. Tank T101 mass balance: FIT101(in) / FIT201(out) ≈ 1.0 normal
    f['bal_T101']    = np.where(fit201 > 0.01, fit101 / (fit201 + 1e-6), 1.0)
    # 2. Tank T301 mass balance
    f['bal_T301']    = np.where(fit401 > 0.01, fit301 / (fit401 + 1e-6), 1.0)
    # 3. Chemical dilution index: AIT202 / FIT201 — constant during normal
    f['chem_dil']    = np.where(fit201 > 0.01, ait202 / (fit201 + 1e-6), 0.0)
    # 4. Chemical sensor ratio: AIT203 / AIT202 — tracks dosing consistency
    f['chem_ratio']  = np.where(ait202 > 0.01, ait203 / (ait202 + 1e-6), 1.0)
    # 5. RO pressure differential: PIT503 - PIT501 (attack → drops)
    f['dp_RO']       = (pit503 - pit501).clip(-500, 500)
    # 6. Tank level rate of change: dLIT101/dt (attack → unbounded)
    f['dlit101']     = lit101.diff().fillna(0).clip(-100, 100)
    # 7. dLIT301/dt
    f['dlit301']     = lit301.diff().fillna(0).clip(-100, 100)
    # 8. dLIT401/dt
    f['dlit401']     = lit401.diff().fillna(0).clip(-100, 100)
    # 9. Inter-stage flow consistency P1→P2: FIT101 - FIT201 ≈ dLIT101/dt
    f['flow_cons12'] = (fit101 - fit201).clip(-500, 500)
    # 10. Inter-stage flow consistency P3→P4
    f['flow_cons34'] = (fit301 - fit401).clip(-500, 500)
    # 11. Pump flow proxy: P101 state × FIT101 flow (attack: pump off but flow nonzero)
    f['pump_flow']   = (p101 * fit101).clip(-500, 500)
    # 12. Valve-flow consistency: valve open ↔ flow positive
    f['valve_flow']  = np.abs(mv101 - (fit101 > 0.1).astype(float))

    return pd.DataFrame(f, index=df.index).fillna(0).clip(-1e4, 1e4)


def get_net_cols(df):
    """Network feature columns = everything that's not a sensor, meta, or component col."""
    known = META_COLS | set(CONT_SENSORS) | set(BIN_SENSORS)
    return [c for c in df.columns
            if c not in known and not c.startswith(COMP_PREFIX)]


def load_csv(path):
    """Load merged CSV, parse timestamps, normalise column names, forward-fill short gaps."""
    print(f'    Reading {path}...')
    df = pd.read_csv(path, low_memory=False)
    df['timestamp_sgt'] = pd.to_datetime(df['timestamp_sgt'], errors='coerce')
    df = df.dropna(subset=['timestamp_sgt']).sort_values('timestamp_sgt').reset_index(drop=True)
    df = normalise_sensor_cols(df)   # strips .Pv/.Status suffixes, encodes .Alarm -> 0/1
    # Force all non-timestamp, non-string columns to float (catches any leftover object dtype)
    for col in df.columns:
        if col != 'timestamp_sgt' and col != 'attack_name':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill(limit=5)           # forward-fill gaps up to 5s (network hiccups)
    print(f'    Shape after normalise: {df.shape}')
    return df


def make_windows(arr, win, stride):
    """Sliding windows: (N, D) → (M, win, D)."""
    N = len(arr)
    idx = np.arange(0, N - win + 1, stride)
    return np.stack([arr[i:i+win] for i in idx]).astype(np.float32)


def label_windows(y_bin, y_cls, y_comp, win, stride):
    """Create labels for each window.
    Window is labeled ATTACK if ANY second within it was an attack.
    Class = most common attack class in the window.
    Component = element-wise OR over all seconds.
    """
    N = len(y_bin)
    idxs = list(range(0, N - win + 1, stride))
    M, K = len(idxs), y_comp.shape[1]
    out_b = np.zeros(M, dtype=np.int64)
    out_c = np.zeros(M, dtype=np.int64)
    out_k = np.zeros((M, K), dtype=np.float32)
    for i, s in enumerate(idxs):
        e = s + win
        if y_bin[s:e].max() == 1:
            out_b[i] = 1
            clss = y_cls[s:e][y_cls[s:e] > 0]
            out_c[i] = int(pd.Series(clss).mode()[0]) if len(clss) else 1
        out_k[i] = y_comp[s:e].max(axis=0)
    return out_b, out_c, out_k


# ─── MAIN ─────────────────────────────────────────────────────────

def main():
    Path('trustgate_data').mkdir(exist_ok=True)

    # ── [1] Load A11 (pure normal — fits the scaler) ──────────────
    print('\n[1/6] Loading A11 normal baseline...')
    a11 = load_csv(A11_CSV)
    c_cont = [c for c in CONT_SENSORS if c in a11.columns]
    c_bin  = [c for c in BIN_SENSORS  if c in a11.columns]
    c_net  = get_net_cols(a11)
    print(f'    cont:{len(c_cont)} binary:{len(c_bin)} network:{len(c_net)}')

    # ── [2] Physics features ──────────────────────────────────────
    print('[2/6] Computing physics features on A11...')
    phys_a11 = compute_physics_features(a11)
    phys_cols = list(phys_a11.columns)

    # ── [3] Fit Robust Scaler on A11 ONLY ─────────────────────────
    print('[3/6] Fitting RobustScaler on A11...')
    # Scale continuous sensors + physics features together
    cont_phys_a11 = np.hstack([a11[c_cont].fillna(0).values, phys_a11.values])
    scaler_s = RobustScaler().fit(cont_phys_a11)

    # Network scaler also fit on A11 (z-scores: attack → outlier)
    net_a11_raw = a11[c_net].fillna(0).values if c_net else np.zeros((len(a11), 1))
    scaler_n = RobustScaler().fit(net_a11_raw)

    def to_sensor_array(df, phys):
        cp = np.hstack([df[c_cont].fillna(0).values, phys.values])
        scaled = scaler_s.transform(cp)
        bins   = df[c_bin].fillna(0).values if c_bin else np.zeros((len(df), 0))
        # Layout: [scaled_cont | binary | scaled_physics]
        return np.hstack([scaled[:, :len(c_cont)], bins,
                          scaled[:, len(c_cont):]]).astype(np.float32)

    def to_net_array(df):
        raw = df[c_net].fillna(0).values if c_net else np.zeros((len(df), 1))
        return scaler_n.transform(raw).astype(np.float32)

    Xs_a11 = to_sensor_array(a11, phys_a11)
    Xn_a11 = to_net_array(a11)
    sensor_cols = c_cont + c_bin + phys_cols

    with open(SCALER_PKL, 'wb') as fh:
        pickle.dump({'scaler_s': scaler_s, 'scaler_n': scaler_n,
                     'c_cont': c_cont, 'c_bin': c_bin,
                     'c_net': c_net, 'phys_cols': phys_cols,
                     'sensor_cols': sensor_cols}, fh)
    print(f'    Sensor dim: {Xs_a11.shape[1]}  Net dim: {Xn_a11.shape[1]}')
    print(f'    Saved {SCALER_PKL}')

    # ── [4] Load A12 (with attacks) ───────────────────────────────
    print('[4/6] Loading A12 attack dataset...')
    a12 = load_csv(A12_CSV)
    for col in c_cont + c_bin + c_net:
        if col not in a12.columns:
            a12[col] = 0.0

    c_comp = [c for c in a12.columns if c.startswith(COMP_PREFIX)]
    for col in c_comp:
        if col not in a12.columns:
            a12[col] = 0.0
    if 'label_binary' not in a12.columns: a12['label_binary'] = 0
    if 'label_class'  not in a12.columns: a12['label_class']  = 0

    phys_a12 = compute_physics_features(a12)
    Xs_a12   = to_sensor_array(a12, phys_a12)
    Xn_a12   = to_net_array(a12)
    y_bin    = a12['label_binary'].fillna(0).values.astype(np.int64)
    y_cls    = a12['label_class'].fillna(0).values.astype(np.int64)
    y_comp   = (a12[c_comp].fillna(0).values.astype(np.float32)
                if c_comp else np.zeros((len(a12), 1), dtype=np.float32))
    print(f'    Attack rows: {y_bin.sum():,} / {len(y_bin):,} '
          f'({y_bin.mean()*100:.1f}%)')

    # ── [5] Create windows (per-event chronological attack split) ─
    print('[5/6] Creating windows with per-event chronological split...')
    print('      Normal data: chronological 70/15/15')
    print('      Attack events: each event split 70/15/15 internally with purge buffer')

    # Build per-row split membership (0=train, 1=val, 2=test)
    atk_name_col = a12['attack_name'].values if 'attack_name' in a12.columns                    else np.array(['NORMAL'] * len(a12))

    # Chronological split of NORMAL rows
    normal_idx  = np.where(y_bin == 0)[0]
    N_norm      = len(normal_idx)
    n_tr_norm   = int(N_norm * TRAIN_R)
    n_va_norm   = int(N_norm * VAL_R)
    train_set   = set(normal_idx[:n_tr_norm].tolist())
    val_set     = set(normal_idx[n_tr_norm:n_tr_norm + n_va_norm].tolist())

    row_split = np.full(len(y_bin), -1, dtype=np.int8)
    for i in range(len(y_bin)):
        if y_bin[i] == 0:
            if i in train_set:   row_split[i] = 0
            elif i in val_set:   row_split[i] = 1
            else:                row_split[i] = 2

    # ── Per-event chronological split for ATTACK rows ──────────────
    # Each named attack event gets its own internal 70/15/15 time split,
    # with a purge buffer dropped around each cut so no window straddles
    # a split boundary. This guarantees every attack class appears in
    # every split. Events too short to split safely stay whole in TRAIN.
    attack_event_names = sorted(set(str(n) for n in atk_name_col
                                     if str(n) != 'NORMAL'))
    print(f'    Found {len(attack_event_names)} attack events: {attack_event_names}')

    for name in attack_event_names:
        evt_idx = np.where(atk_name_col == name)[0]
        evt_idx.sort()
        n_evt = len(evt_idx)

        if n_evt < MIN_SPLIT_ROWS:
            # Too short to split safely — keep whole in TRAIN
            row_split[evt_idx] = 0
            print(f'      [{name}] {n_evt} rows — too short to split, kept whole in TRAIN')
            continue

        n_tr = int(n_evt * TRAIN_R)
        n_va = int(n_evt * VAL_R)
        # boundaries within the event's local row sequence
        tr_end  = n_tr
        va_start = tr_end + PURGE_SEC
        va_end   = tr_end + n_va
        te_start = va_end + PURGE_SEC

        tr_local = evt_idx[:max(0, tr_end)]
        va_local = evt_idx[va_start:va_end] if va_start < va_end else np.array([], dtype=int)
        te_local = evt_idx[te_start:] if te_start < n_evt else np.array([], dtype=int)

        row_split[tr_local] = 0
        row_split[va_local] = 1
        row_split[te_local] = 2
        print(f'      [{name}] {n_evt} rows -> train={len(tr_local)} '
              f'val={len(va_local)} test={len(te_local)} (purge={PURGE_SEC}s each cut)')

    # Any leftover unassigned attack rows (shouldn't normally happen) -> train
    row_split[(row_split == -1) & (y_bin == 1)] = 0

    split_map = {0: 'train', 1: 'val', 2: 'test'}
    for sid, sname in split_map.items():
        m = row_split == sid
        print(f'    {sname}: {m.sum():,} rows  '
              f'({int(y_bin[m].sum())} attack  {int((m.sum()-y_bin[m].sum()))} normal)')

    # Per-class sanity check: confirm every attack class reaches every split
    print('    Per-split class coverage check:')
    for sid, sname in split_map.items():
        m = row_split == sid
        classes_present = sorted(set(y_cls[m][y_bin[m] == 1].tolist())) if m.sum() else []
        print(f'      {sname}: attack classes present = {classes_present}')

    # Build windows: only windows where ALL rows belong to the same split are kept
    N = len(Xs_a12)
    win_data = {}
    for sid, tag in split_map.items():
        for win, stride, suffix in [
            (W_STD,   STRIDE,       ''),
            (W_MICRO, STRIDE,       '_micro'),
            (W_MACRO, STRIDE_MACRO, '_macro'),
        ]:
            xs_list, xn_list, yb_list, yc_list, yk_list, name_list = [], [], [], [], [], []
            i = 0
            while i <= N - win:
                if np.all(row_split[i:i+win] == sid):
                    xs_list.append(Xs_a12[i:i+win])
                    xn_list.append(Xn_a12[i:i+win])
                    yb_list.append(y_bin[i:i+win])
                    yc_list.append(y_cls[i:i+win])
                    yk_list.append(y_comp[i:i+win])
                    # Store most common attack name in this window
                    window_names = atk_name_col[i:i+win]
                    atk_names_in_win = [n for n in window_names if n != 'NORMAL']
                    name_list.append(atk_names_in_win[0] if atk_names_in_win else 'NORMAL')
                    i += stride
                else:
                    i += 1
            if not xs_list:
                continue
            xsw = np.stack(xs_list).astype(np.float32)
            xnw = np.stack(xn_list).astype(np.float32)
            yb_arr = np.zeros(len(yb_list), dtype=np.int64)
            yc_arr = np.zeros(len(yc_list), dtype=np.int64)
            yk_arr = np.zeros((len(yk_list), y_comp.shape[1]), dtype=np.float32)
            for wi in range(len(yb_list)):
                if yb_list[wi].max() == 1:
                    yb_arr[wi] = 1
                    clss = yc_list[wi][yc_list[wi] > 0]
                    yc_arr[wi] = int(pd.Series(clss).mode()[0]) if len(clss) else 1
                yk_arr[wi] = yk_list[wi].max(axis=0)
            name_arr = np.array(name_list, dtype=object)
            key = f'{tag}{suffix}'
            win_data[key] = (xsw, xnw, yb_arr, yc_arr, yk_arr, name_arr)
            print(f'    {key:20s}: {xsw.shape[0]:>7,} windows  '
                  f'({yb_arr.mean()*100:.1f}% attack)  shape={xsw.shape}')


    # A11 pre-training windows (standard size, all normal)
    Xs_pre = make_windows(Xs_a11, W_STD, STRIDE)
    Xn_pre = make_windows(Xn_a11, W_STD, STRIDE)
    print(f'    {"pretrain":20s}: {Xs_pre.shape[0]:>7,} windows  (0.0% attack)  shape={Xs_pre.shape}')

    # ── [6] Save ──────────────────────────────────────────────────
    print(f'[6/6] Saving {OUT_NPZ}...')
    save_dict = dict(
        sensor_cols=np.array(sensor_cols),
        net_cols   =np.array(c_net if c_net else ['no_net']),
        comp_cols  =np.array(c_comp if c_comp else ['no_comp']),
        X_s_pretrain=Xs_pre, X_n_pretrain=Xn_pre,
    )
    for tag, (xsw, xnw, yb_, yc_, yk_, names_) in win_data.items():
        # tag examples: 'train', 'val', 'test', 'train_micro', 'test_macro', ...
        save_dict[f'X_s_{tag}'] = xsw
        save_dict[f'X_n_{tag}'] = xnw
        save_dict[f'y_{tag}']   = yb_
        if '_' not in tag:  # only add class/comp for standard windows
            save_dict[f'y_class_{tag}']  = yc_
            save_dict[f'y_comp_{tag}']   = yk_
            save_dict[f'y_names_{tag}']  = names_  # attack name per window

    np.savez_compressed(OUT_NPZ, **save_dict)
    sz = Path(OUT_NPZ).stat().st_size / 1e6
    print(f'    Saved: {OUT_NPZ}  ({sz:.1f} MB)')
    print('\n[DONE] Run python create_labels.py  →  then 02_granger_causality.py')


if __name__ == '__main__':
    main()