"""
build_traffic.py
----------------
Preprocess METR-LA and PEMS-BAY traffic speed datasets.

Standard benchmarks from DCRNN (Li et al., 2018), used by virtually every
spatio-temporal forecasting paper.

Expected raw files per dataset
──────────────────────────────
METR-LA  →  data/metr_la/raw/
  metr-la.csv          OR  metr-la.h5   (speed, shape T×207)
  adj_mx_METR-LA.pkl                    (DCRNN adjacency pickle)

PEMS-BAY →  data/pems_bay/raw/
  pems-bay.csv         OR  pems-bay.h5  (speed, shape T×325)
  adj_mx_PEMS-BAY.pkl                   (DCRNN adjacency pickle)

CSV format (confirmed from sample files):
  - First column: datetime index  (e.g. "3/1/2012 0:00")
  - Remaining columns: sensor IDs as headers  (e.g. "773869")
  - Values: speed in mph at 5-minute intervals
  - METR-LA zeros → sensor failure → treated as NaN

Adjacency pickle format (DCRNN):
  [sensor_ids: list[str],
   sensor_id_to_idx: dict[str, int],
   adj_matrix: ndarray (N, N)]
  Values are Gaussian-kernel road-distance weights in [0, 1].
  Matrix is directed (asymmetric). Diagonal = 1.0.

Temporal resolution
───────────────────
We keep native 5-min resolution:
  12 steps = 60 minutes  (history window and forecast horizon)
  This is the standard traffic forecasting setup.

This is deliberately different from air quality (12 steps × 1h = 12h).
The difference is noted in Section 5.1 of the paper.

Processing steps
────────────────
  1. Load CSV (or H5) speed data
  2. Align columns to adjacency sensor order
  3. Convert zeros to NaN (METR-LA sensor failures)
  4. Save data_raw.npy  ← before imputation, used by data_audit.py
  5. Impute with 1-hour centred rolling mean (12 × 5-min steps)
  6. Process adjacency: enforce symmetry, remove self-loops
  7. Compute normalised Laplacian + eigendecomposition
  8. Normalise (global mean / std)
  9. Split 70/10/20  (DCRNN convention)
 10. Save all artefacts

Outputs  data/{dataset}/processed/
  data.npy            (T, N, 1)  normalised speed
  data_raw.npy        (T, N, 1)  raw speed, NaN preserved (for audit)
  adj.npy             (N, N)
  laplacian.npy       (N, N)
  eigenvalues.npy     (N,)
  eigenvectors.npy    (N, N)
  mean.npy / std.npy  (1,)
  splits.npz          train / val / test index arrays
  station_info.pkl    metadata dict
"""

import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from graph_utils import (
    normalized_laplacian, compute_eigen,
    validate_adjacency, validate_laplacian,
)

# ── dataset configs ───────────────────────────────────────────────────────────

CONFIGS = {
    'metr_la': {
        'raw_dir':      'data/metr_la/raw',
        'out_dir':      'data/metr_la/processed',
        # CSV filename candidates (first found is used)
        'csv_names':    ['metr-la.csv', 'METR-LA.csv', 'metr_la.csv'],
        'h5_names':     ['metr-la.h5', 'metr_la.h5'],
        'h5_key':       'df',
        # Adjacency filename candidates
        'adj_names':    ['adj_mx_METR-LA.pkl', 'adj_mx.pkl', 'adj_mx_metr_la.pkl'],
        'n_sensors':    207,
        'freq_min':     5,
        'zero_is_nan':  True,    # zeros = sensor failure in METR-LA
        'split':        (0.7, 0.1, 0.2),
        'feature':      'speed_mph',
    },
    'pems_bay': {
        'raw_dir':      'data/pems_bay/raw',
        'out_dir':      'data/pems_bay/processed',
        'csv_names':    ['pems-bay.csv', 'PEMS-BAY.csv', 'pems_bay.csv'],
        'h5_names':     ['pems-bay.h5', 'pems_bay.h5'],
        'h5_key':       'df',
        'adj_names':    ['adj_mx_PEMS-BAY.pkl', 'adj_mx_bay.pkl', 'adj_mx_pems_bay.pkl'],
        'n_sensors':    325,
        'freq_min':     5,
        'zero_is_nan':  False,   # PEMS-BAY zeros are genuine slow speeds
        'split':        (0.7, 0.1, 0.2),
        'feature':      'speed_mph',
    },
}

# ── loaders ───────────────────────────────────────────────────────────────────

def _find_file(directory: str, candidates: list) -> str:
    """Return first existing file from candidates list, or raise."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"None of {candidates} found in {directory}\n"
        f"Files present: {os.listdir(directory) if os.path.isdir(directory) else '(dir not found)'}"
    )


def load_speed_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CSV with format:
      col 0 : datetime string  ('3/1/2012 0:00')
      col 1+: sensor IDs as headers, speed values in mph

    Handles:
      - Mixed dtype warnings (some rows may have non-numeric text)
      - Parse errors in the datetime column
    Returns DataFrame with DatetimeIndex, columns = sensor ID strings.
    """
    print(f"  Reading {os.path.basename(csv_path)} ...", flush=True)
    df = pd.read_csv(csv_path, index_col=0, low_memory=False)

    # Parse datetime index
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        print(f"  [WARN] Datetime parse failed ({e}), creating synthetic index")
        freq = '5T'
        df.index = pd.date_range(start='2012-01-01', periods=len(df), freq=freq)

    # Force all columns to numeric (some rows may have header artefacts)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop any rows that are entirely NaN (padding rows at end of CSV sample)
    df = df.dropna(how='all')

    # Ensure column names are strings
    df.columns = df.columns.astype(str)

    print(f"  Loaded: {df.shape}  [{df.index[0]} → {df.index[-1]}]")
    return df


def load_speed_h5(h5_path: str, h5_key: str) -> pd.DataFrame:
    """Load speed from HDF5 file."""
    try:
        df = pd.read_hdf(h5_path, key=h5_key)
        df.columns = df.columns.astype(str)
        print(f"  Loaded H5: {df.shape}")
        return df
    except Exception as e:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            print(f"  H5 keys: {list(f.keys())}")
            arr = f[list(f.keys())[0]][:]
        df = pd.DataFrame(arr)
        print(f"  Loaded H5 (h5py): {df.shape}")
        return df


def load_adjacency(pkl_path: str) -> tuple:
    """
    Load DCRNN adjacency pickle.
    Format: [sensor_ids, sensor_id_to_idx, adj_matrix]
    Returns (sensor_ids: list[str], sensor_id_to_idx: dict, adj_matrix: ndarray)
    """
    with open(pkl_path, 'rb') as f:
        try:
            contents = pickle.load(f, encoding='latin1')
        except TypeError:
            contents = pickle.load(f)

    sensor_ids, sensor_id_to_idx, adj_matrix = contents
    sensor_ids = [str(s) for s in sensor_ids]    # ensure string
    sensor_id_to_idx = {str(k): v for k, v in sensor_id_to_idx.items()}

    print(f"  Adjacency: {len(sensor_ids)} sensors, matrix {adj_matrix.shape}")
    print(f"  Non-zero entries: {(adj_matrix > 0).sum()} "
          f"(sparsity {(adj_matrix == 0).mean()*100:.1f}%)")
    print(f"  Directed: {not np.allclose(adj_matrix, adj_matrix.T)}")
    return sensor_ids, sensor_id_to_idx, adj_matrix.astype(np.float32)


# ── imputation ────────────────────────────────────────────────────────────────

def impute_rolling_mean(arr: np.ndarray, window: int = 12) -> np.ndarray:
    """
    Impute NaN with centered rolling mean (window in timesteps).
    Remaining NaN → column global mean → 0.

    arr: (T, N) or (T, N, C)
    """
    original_shape = arr.shape
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    T, N, C = arr.shape
    out = arr.copy()

    for n in range(N):
        for c in range(C):
            series = pd.Series(out[:, n, c])
            filled = series.fillna(
                series.rolling(window, min_periods=1, center=True).mean()
            )
            global_mean = filled.mean()
            filled = filled.fillna(0.0 if pd.isna(global_mean) else global_mean)
            out[:, n, c] = filled.values

    return out.reshape(original_shape)


# ── main preprocessing ────────────────────────────────────────────────────────

def preprocess(dataset_name: str, base_dir: str = None):
    if dataset_name not in CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(CONFIGS)}")

    cfg     = CONFIGS[dataset_name]
    root    = base_dir or os.path.join(os.path.dirname(__file__), '..')
    raw_dir = os.path.join(root, cfg['raw_dir'])
    out_dir = os.path.join(root, cfg['out_dir'])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print("\n" + "=" * 64)
    print(f"  Building {dataset_name.upper()}")
    print("=" * 64)

    # ── 1. Load speed data ──────────────────────────────────────────
    print("\n[1] Loading speed data...")

    # Try CSV first, then H5
    speed_df = None
    try:
        csv_path = _find_file(raw_dir, cfg['csv_names'])
        speed_df = load_speed_csv(csv_path)
    except FileNotFoundError:
        pass

    if speed_df is None:
        try:
            h5_path  = _find_file(raw_dir, cfg['h5_names'])
            speed_df = load_speed_h5(h5_path, cfg['h5_key'])
        except FileNotFoundError:
            pass

    if speed_df is None:
        raise FileNotFoundError(
            f"\nCould not find speed data in {raw_dir}\n"
            f"Expected one of: {cfg['csv_names'] + cfg['h5_names']}\n"
            f"Files present: {os.listdir(raw_dir)}"
        )

    # ── 2. Load adjacency ────────────────────────────────────────────
    print("\n[2] Loading adjacency pickle...")
    adj_path = _find_file(raw_dir, cfg['adj_names'])
    sensor_ids, sensor_id_to_idx, adj_raw = load_adjacency(adj_path)
    N = len(sensor_ids)

    # ── 3. Align columns to sensor order from pickle ─────────────────
    print("\n[3] Aligning sensor order...")
    csv_sensors = set(speed_df.columns.astype(str))
    pkl_sensors = set(sensor_ids)

    overlap = csv_sensors & pkl_sensors
    if len(overlap) < N:
        print(f"  [WARN] Only {len(overlap)}/{N} sensors overlap CSV↔pickle")
        # Use sensors that appear in both, in pickle order
        sensor_ids = [s for s in sensor_ids if s in csv_sensors]
        N = len(sensor_ids)
    else:
        print(f"  ✓ All {N} sensors matched")

    # Reorder columns to match pickle order
    speed_df = speed_df[[str(s) for s in sensor_ids]]

    T = len(speed_df)
    print(f"  Final shape: ({T}, {N})")
    print(f"  Time: {speed_df.index[0]} → {speed_df.index[-1]}")
    print(f"  Timestep: {cfg['freq_min']} min → "
          f"12 steps = {12 * cfg['freq_min']} min forecast horizon")

    # ── 4. Convert zeros to NaN if applicable ────────────────────────
    data = speed_df.values.astype(np.float32)  # (T, N)

    if cfg['zero_is_nan']:
        n_zeros = (data == 0).sum()
        if n_zeros > 0:
            print(f"\n[4] Converting {n_zeros} zeros to NaN (METR-LA sensor failures)...")
            data = np.where(data == 0, np.nan, data)
        else:
            print("\n[4] No zeros found.")

    nan_frac_raw = np.isnan(data).mean()
    print(f"  NaN fraction before imputation: {nan_frac_raw*100:.3f}%")

    # ── 5. Save raw data (before imputation, for audit) ──────────────
    data_3d = data[:, :, np.newaxis]  # (T, N, 1)
    np.save(os.path.join(out_dir, 'data_raw.npy'), data_3d.astype(np.float32))
    print(f"  Saved data_raw.npy (pre-imputation)")

    # ── 6. Impute ────────────────────────────────────────────────────
    print("\n[5] Imputing missing values (12-step centred rolling mean = 1h window)...")
    data_3d = impute_rolling_mean(data_3d, window=12)
    nan_after = np.isnan(data_3d).mean()
    if nan_after > 0:
        print(f"  [WARN] {nan_after*100:.4f}% NaN remain after imputation → filling with 0")
        data_3d = np.nan_to_num(data_3d, nan=0.0)
    else:
        print(f"  ✓ Zero NaN remaining")

    # ── 7. Adjacency processing ───────────────────────────────────────
    print("\n[6] Processing adjacency matrix...")
    W = adj_raw[:N, :N].copy()        # slice if sensors were dropped

    # Remove self-loops (diagonal = 1.0 in source pickle)
    np.fill_diagonal(W, 0.0)

    # Symmetrise: average of W and W^T
    # (directed graph → bidirectional weights)
    W_sym = (W + W.T) / 2.0
    print(f"  Symmetrised (was directed). Non-zero: {(W_sym > 0).sum()}")

    validate_adjacency(W_sym, name=f"{dataset_name} adj")

    # ── 8. Laplacian + eigendecomposition ────────────────────────────
    print("\n[7] Computing normalised Laplacian and eigendecomposition...")
    L = normalized_laplacian(W_sym)
    validate_laplacian(L, name=f"{dataset_name} L")

    cache = os.path.join(out_dir, 'eigen_cache.pkl')
    eigenvalues, eigenvectors = compute_eigen(L, cache_path=cache)

    # ── 9. Normalise ─────────────────────────────────────────────────
    print("\n[8] Normalising...")
    mean_val = np.nanmean(data_3d, axis=(0, 1), keepdims=False)  # (1,)
    std_val  = np.nanstd(data_3d,  axis=(0, 1), keepdims=False)
    std_val  = np.where(std_val < 1e-6, 1.0, std_val)

    data_norm = (data_3d - mean_val) / std_val
    print(f"  Speed: mean={mean_val[0]:.2f} mph, std={std_val[0]:.2f}")

    # ── 10. Train/Val/Test split ──────────────────────────────────────
    split  = cfg['split']
    n_train = int(T * split[0])
    n_val   = int(T * split[1])
    n_test  = T - n_train - n_val
    print(f"\n[9] Split (70/10/20): Train={n_train} | Val={n_val} | Test={n_test} steps")
    print(f"  ≈ {n_train*cfg['freq_min']//60:.0f}h / {n_val*cfg['freq_min']//60:.0f}h / "
          f"{n_test*cfg['freq_min']//60:.0f}h")

    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, T)

    # ── 11. Save all artefacts ────────────────────────────────────────
    print("\n[10] Saving processed files...")
    np.save(os.path.join(out_dir, 'data.npy'),         data_norm.astype(np.float32))
    np.save(os.path.join(out_dir, 'adj.npy'),          W_sym.astype(np.float32))
    np.save(os.path.join(out_dir, 'laplacian.npy'),    L.astype(np.float32))
    np.save(os.path.join(out_dir, 'eigenvalues.npy'),  eigenvalues.astype(np.float32))
    np.save(os.path.join(out_dir, 'eigenvectors.npy'), eigenvectors.astype(np.float32))
    np.save(os.path.join(out_dir, 'mean.npy'),         mean_val.astype(np.float32))
    np.save(os.path.join(out_dir, 'std.npy'),          std_val.astype(np.float32))
    np.savez(os.path.join(out_dir, 'splits.npz'),
             train=train_idx, val=val_idx, test=test_idx)

    station_info = {
        'sensor_ids':        sensor_ids,
        'n_stations':        N,
        'n_timesteps':       T,
        'freq_min':          cfg['freq_min'],
        'feature':           cfg['feature'],
        'has_wind':          False,
        'pollutants':        [cfg['feature']],   # unified interface
        'zero_was_nan':      cfg['zero_is_nan'],
        'nan_frac_raw':      float(nan_frac_raw),
    }
    with open(os.path.join(out_dir, 'station_info.pkl'), 'wb') as fh:
        pickle.dump(station_info, fh)

    # Summary
    print(f"\n{'─'*64}")
    print(f"  ✓  {dataset_name.upper()} complete")
    print(f"     data.npy       : {data_norm.shape}  (T={T}, N={N}, C=1)")
    print(f"     Speed range    : {data_3d.min():.1f} – {data_3d.max():.1f} mph (raw)")
    print(f"     Missing raw    : {nan_frac_raw*100:.2f}%")
    print(f"     Adj non-zero   : {(W_sym>0).sum()} edges")
    print(f"     Output dir     : {out_dir}")
    print(f"{'─'*64}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess METR-LA and/or PEMS-BAY traffic datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preprocessing/build_traffic.py metr_la
  python preprocessing/build_traffic.py pems_bay
  python preprocessing/build_traffic.py both

Raw files expected in:
  data/metr_la/raw/   → metr-la.csv (or .h5)   + adj_mx_METR-LA.pkl
  data/pems_bay/raw/  → pems-bay.csv (or .h5)  + adj_mx_PEMS-BAY.pkl
"""
    )
    parser.add_argument('dataset', choices=['metr_la', 'pems_bay', 'both'],
                        help='Dataset to preprocess')
    parser.add_argument('--base_dir', default=None,
                        help='Project root directory (default: parent of this script)')
    args = parser.parse_args()

    datasets = ['metr_la', 'pems_bay'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets:
        preprocess(ds, base_dir=args.base_dir)
