"""
build_athens.py
---------------
Preprocess the Athens air quality dataset.

Input files (place in data/athens/raw/):
  athens_aq.csv    - hourly readings with lat/lon and wind U/V per station

Column layout (from sample):
  Date, Latitude, Longitude, station_name,
  Wind-Speed (U), Wind-Speed (V), Dewpoint Temp, Soil Temp,
  Total Precipitation, Vegetation (High), Vegetation (Low),
  Temp, Relative Humidity, PM10, PM2.5, NO2, O3, code, id

Key differences vs Beijing:
  - Single file: pollutants, coordinates, AND wind components all present
  - Wind is already in U/V form (no trigonometric conversion needed)
  - Pollutants: PM2.5, PM10, NO2, O3 (no CO, SO2)
  - Mix of CAMS grid stations and named monitoring stations

Processing steps:
  1. Load CSV, parse datetime
  2. Identify and deduplicate stations
  3. Pivot to (T, N, C) array
  4. Build distance-based adjacency from lat/lon
  5. Compute normalized Laplacian and eigendecomposition
  6. Extract wind U/V arrays (per station, aligned to same time index)
  7. Normalize and split

Outputs (data/athens/processed/):
  Same structure as Beijing processed output.
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from graph_utils import (
    build_adjacency, normalized_laplacian, compute_eigen,
    validate_adjacency, validate_laplacian,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'athens', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'athens', 'processed')

# Expected filename (user drops this in raw/)
AQ_FILE = 'athens_aq.csv'

# Pollutants to extract (in order → channel indices)
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'O3']

# Column name mapping (raw CSV → standard names)
COL_MAP = {
    'Date':              'utc_time',
    'Latitude':          'latitude',
    'Longitude':         'longitude',
    'station_name':      'stationId',
    'Wind-Speed (U)':    'wind_u',
    'Wind-Speed (V)':    'wind_v',
    'PM10':              'PM10',
    'PM2.5':             'PM2.5',
    'NO2':               'NO2',
    'O3':                'O3',
}

SPLIT = (0.6, 0.2, 0.2)
SIGMA = None
ADJ_THRESHOLD = 0.1


def load_data(raw_dir: str) -> pd.DataFrame:
    """Load Athens CSV, rename columns, parse datetime."""
    fpath = os.path.join(raw_dir, AQ_FILE)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Athens data file not found: {fpath}\n"
            f"Place '{AQ_FILE}' in {raw_dir}"
        )

    df = pd.read_csv(fpath, encoding='utf-8-sig')  # utf-8-sig handles BOM
    # Rename columns we care about
    rename = {k: v for k, v in COL_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    print(f"  [Athens] Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    return df


def get_station_coords(df: pd.DataFrame) -> dict:
    """
    Extract {station_id: (lat, lon)} from the dataframe.
    Use the first occurrence of each station (coordinates are static).
    """
    coords = (
        df.groupby('stationId')[['latitude', 'longitude']]
        .first()
        .to_dict('index')
    )
    return {k: (v['latitude'], v['longitude']) for k, v in coords.items()}


def pivot_to_array(
    df: pd.DataFrame,
    stations: list,
    time_index: pd.DatetimeIndex,
    columns: list,
) -> np.ndarray:
    """Pivot long-format df to (T, N, C) array."""
    T = len(time_index)
    N = len(stations)
    C = len(columns)
    arr = np.full((T, N, C), np.nan, dtype=np.float32)

    df_idx = df.set_index(['utc_time', 'stationId'])

    for n, station in enumerate(stations):
        if station not in df_idx.index.get_level_values('stationId'):
            continue
        sub = df_idx.xs(station, level='stationId')
        sub = sub.reindex(time_index)
        for c, col in enumerate(columns):
            if col in sub.columns:
                arr[:, n, c] = sub[col].values.astype(np.float32)

    return arr


def build_wind_arrays(
    df: pd.DataFrame,
    stations: list,
    time_index: pd.DatetimeIndex,
) -> tuple:
    """
    Build wind U and V arrays from the dataframe.
    Athens already provides U and V components directly.
    """
    if 'wind_u' not in df.columns or 'wind_v' not in df.columns:
        print("  [WARNING] wind_u/wind_v columns not found. Wind features unavailable.")
        return None, None

    wind_u = pivot_to_array(df, stations, time_index, ['wind_u'])[:, :, 0]
    wind_v = pivot_to_array(df, stations, time_index, ['wind_v'])[:, :, 0]
    return wind_u, wind_v


def impute_rolling_mean(arr: np.ndarray, window: int = 24) -> np.ndarray:
    """Impute NaN with centered rolling mean. arr: (T, N) or (T, N, C)."""
    shape = arr.shape
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    T, N, C = arr.shape
    out = arr.copy()
    for n in range(N):
        for c in range(C):
            series = pd.Series(out[:, n, c])
            filled = series.fillna(series.rolling(window, min_periods=1, center=True).mean())
            global_mean = filled.mean()
            if pd.isna(global_mean):
                global_mean = 0.0
            filled = filled.fillna(global_mean)
            out[:, n, c] = filled.values
    out = out.reshape(shape)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n" + "="*60)
    print("Building Athens dataset")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading data...")
    df = load_data(RAW_DIR)

    stations = sorted(df['stationId'].unique().tolist())
    N = len(stations)
    print(f"  Stations ({N}): {stations}")

    # ------------------------------------------------------------------
    # 2. Time index
    # ------------------------------------------------------------------
    t_min = df['utc_time'].min().floor('H')
    t_max = df['utc_time'].max().ceil('H')
    time_index = pd.date_range(t_min, t_max, freq='H')
    T = len(time_index)
    print(f"\n[2] Time index: {t_min} → {t_max} ({T} steps)")

    # ------------------------------------------------------------------
    # 3. Station coordinates
    # ------------------------------------------------------------------
    coords = get_station_coords(df)
    lats = np.array([coords[s][0] for s in stations])
    lons = np.array([coords[s][1] for s in stations])

    # ------------------------------------------------------------------
    # 4. Pivot pollutant data to (T, N, C)
    # ------------------------------------------------------------------
    print("\n[3] Pivoting pollutant data...")
    data = pivot_to_array(df, stations, time_index, POLLUTANTS)
    print(f"  Raw array shape: {data.shape}  NaN fraction: {np.isnan(data).mean():.4f}")

    # ------------------------------------------------------------------
    # 5. Impute
    # ------------------------------------------------------------------
    print("\n[4] Imputing missing values...")
    data = impute_rolling_mean(data, window=24)
    print(f"  After imputation NaN fraction: {np.isnan(data).mean():.6f}")

    # ------------------------------------------------------------------
    # 6. Adjacency + Laplacian + Eigendecomposition
    # ------------------------------------------------------------------
    print("\n[5] Building graph...")
    W = build_adjacency(lats, lons, sigma=SIGMA, threshold=ADJ_THRESHOLD)
    validate_adjacency(W, name="Athens adj")

    L = normalized_laplacian(W)
    validate_laplacian(L, name="Athens L")

    cache = os.path.join(OUT_DIR, 'eigen_cache.pkl')
    eigenvalues, eigenvectors = compute_eigen(L, cache_path=cache)

    # ------------------------------------------------------------------
    # 7. Wind U/V features
    # ------------------------------------------------------------------
    print("\n[6] Building wind features...")
    wind_u, wind_v = build_wind_arrays(df, stations, time_index)

    if wind_u is not None:
        wind_u = impute_rolling_mean(wind_u, window=24)
        wind_v = impute_rolling_mean(wind_v, window=24)
        print(f"  Wind arrays: {wind_u.shape}, NaN: {np.isnan(wind_u).mean():.6f}")
        print("  ✓ Wind U/V components available (ideal for full AirPhyNet advection)")

    # ------------------------------------------------------------------
    # 8. Normalize
    # ------------------------------------------------------------------
    print("\n[7] Normalizing...")
    mean = data.mean(axis=(0, 1))
    std  = data.std(axis=(0, 1))
    std  = np.where(std < 1e-6, 1.0, std)
    data_norm = (data - mean) / std
    print(f"  Means: {dict(zip(POLLUTANTS, mean.round(3)))}")

    # ------------------------------------------------------------------
    # 9. Split
    # ------------------------------------------------------------------
    print("\n[8] Splitting (6:2:2)...")
    n_train = int(T * SPLIT[0])
    n_val   = int(T * SPLIT[1])
    n_test  = T - n_train - n_val
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, T)
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    print("\n[9] Saving...")
    np.save(os.path.join(OUT_DIR, 'data.npy'),         data_norm.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'data_raw.npy'),     data.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'adj.npy'),          W.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'laplacian.npy'),    L.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'eigenvalues.npy'),  eigenvalues.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'eigenvectors.npy'), eigenvectors.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'mean.npy'),         mean.astype(np.float32))
    np.save(os.path.join(OUT_DIR, 'std.npy'),          std.astype(np.float32))
    np.savez(os.path.join(OUT_DIR, 'splits.npz'),
             train=train_idx, val=val_idx, test=test_idx)

    if wind_u is not None:
        np.save(os.path.join(OUT_DIR, 'wind_u.npy'), wind_u.astype(np.float32))
        np.save(os.path.join(OUT_DIR, 'wind_v.npy'), wind_v.astype(np.float32))

    station_info = {
        'stations':    stations,
        'lats':        lats.tolist(),
        'lons':        lons.tolist(),
        'pollutants':  POLLUTANTS,
        'time_index':  time_index,
        'n_stations':  N,
        'n_timesteps': T,
        'has_wind':    wind_u is not None,
    }
    with open(os.path.join(OUT_DIR, 'station_info.pkl'), 'wb') as f:
        pickle.dump(station_info, f)

    print(f"\n✓ Athens preprocessing complete.")
    print(f"  data.npy: {data_norm.shape}  (T={T}, N={N}, C={len(POLLUTANTS)})")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == '__main__':
    main()
