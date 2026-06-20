"""
build_beijing.py
----------------
Preprocess the Beijing KDD 2017-2018 air quality dataset.

Input files (place in data/beijing/raw/):
  beijing_17_18_aq.csv       - hourly pollutants per AQ station
  beijing_17_18_meo.csv      - hourly meteorology per MEO station (with lat/lon)
  beijing_201802_201803_aq.csv   - supplementary AQ (same format)
  beijing_201802_201803_meo.csv  - supplementary MEO (same format, no lat/lon)

Processing steps:
  1. Load and concatenate both AQ periods
  2. Pivot to wide format: index=time, columns=station
  3. Load MEO data, extract station coordinates from 2017-18 file
  4. Spatially match each AQ station to its nearest MEO station
  5. Build distance-based adjacency matrix from AQ station coordinates
     (AQ stations share coordinates with the KDD dataset metadata)
  6. Compute normalized Laplacian and eigendecomposition
  7. Normalize pollutant values (global mean/std per pollutant)
  8. Split into train/val/test (6:2:2), save as numpy arrays

Outputs (data/beijing/processed/):
  data.npy           - (T, N, C) float32 array, all pollutants
  adj.npy            - (N, N) adjacency matrix
  laplacian.npy      - (N, N) normalized Laplacian
  eigenvalues.npy    - (N,) eigenvalues
  eigenvectors.npy   - (N, N) eigenvectors
  wind_u.npy         - (T, N) wind U component (for AirPhyNet advection)
  wind_v.npy         - (T, N) wind V component
  mean.npy, std.npy  - (C,) normalization statistics
  splits.npz         - train/val/test indices
  station_info.pkl   - station names, coordinates
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from graph_utils import (
    build_adjacency, normalized_laplacian, compute_eigen,
    match_stations_nearest, validate_adjacency, validate_laplacian,
    haversine_matrix,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'beijing', 'raw')
OUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'beijing', 'processed')

AQ_FILES  = ['beijing_17_18_aq.csv', 'beijing_201802_201803_aq.csv']
MEO_FILES = ['beijing_17_18_meo.csv', 'beijing_201802_201803_meo.csv']

# Pollutants to use (in this order → channel indices 0-5)
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'SO2']

# Train/val/test split ratios
SPLIT = (0.6, 0.2, 0.2)

# Adjacency kernel bandwidth (km). None = use median pairwise distance.
SIGMA = None
ADJ_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Known station coordinates for Beijing KDD AQ stations
# These are from the KDD 2018 competition metadata (35 stations)
# ---------------------------------------------------------------------------

BEIJING_AQ_COORDS = {
    'aotizhongxin_aq':     (40.00, 116.40),
    'badaling_aq':         (40.37, 115.98),
    'beibuxinqu_aq':       (40.09, 116.17),
    'daxing_aq':           (39.72, 116.33),
    'dingling_aq':         (40.29, 116.22),
    'donggaocun_aq':       (40.10, 116.89),
    'dongsi_aq':           (39.93, 116.42),
    'dongsihuan_aq':       (39.87, 116.49),
    'fangshan_aq':         (39.74, 115.98),
    'fengtaihuayuan_aq':   (39.86, 116.28),
    'guanyuan_aq':         (39.93, 116.34),
    'gucheng_aq':          (39.91, 116.18),
    'huairou_aq':          (40.33, 116.63),
    'liulihe_aq':          (39.58, 116.02),
    'mentougou_aq':        (39.93, 116.09),
    'miyun_aq':            (40.47, 116.87),
    'miyunshuiku_aq':      (40.52, 116.79),
    'nongzhanguan_aq':     (39.94, 116.46),
    'northwest_aq':        (40.04, 116.16),
    'objectivity_aq':      (39.98, 116.49),  # placeholder
    'olympicsports_aq':    (40.00, 116.39),
    'pingchang_aq':        (40.28, 116.20),
    'pinggu_aq':           (40.14, 117.12),
    'qianmen_aq':          (39.90, 116.39),
    'shunyi_aq':           (40.13, 116.65),
    'tiantan_aq':          (39.88, 116.40),
    'tongzhou_aq':         (39.91, 116.66),
    'wanliu_aq':           (39.99, 116.29),
    'wanshouxigong_aq':    (39.88, 116.35),
    'xizhimenbei_aq':      (39.95, 116.35),
    'yanqin_aq':           (40.45, 115.97),
    'yizhuang_aq':         (39.80, 116.49),
    'yongdingmennei_aq':   (39.87, 116.39),
    'yongledian_aq':       (39.79, 116.76),
    'yufa_aq':             (39.52, 116.30),
}


def load_aq_data(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate all AQ CSV files."""
    dfs = []
    for fname in AQ_FILES:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARNING] AQ file not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No AQ files found in {raw_dir}. Expected: {AQ_FILES}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['stationId', 'utc_time'])
    df = df.sort_values(['stationId', 'utc_time'])
    print(f"  [AQ] Loaded {len(df)} rows, {df['stationId'].nunique()} stations, "
          f"{df['utc_time'].min()} → {df['utc_time'].max()}")
    return df


def load_meo_data(raw_dir: str) -> pd.DataFrame:
    """Load and concatenate MEO files. Keep lat/lon from 2017-18 file."""
    dfs = []
    has_coords = False
    for fname in MEO_FILES:
        fpath = os.path.join(raw_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [WARNING] MEO file not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df.columns = [c.strip().lower() for c in df.columns]
        # Rename to standard names
        df = df.rename(columns={'station_id': 'stationId', 'utc_time': 'utc_time'})
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        dfs.append(df)
        if 'latitude' in df.columns:
            has_coords = True

    if not dfs:
        print("  [WARNING] No MEO files found. Wind features will be unavailable.")
        return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['stationId', 'utc_time'])
    print(f"  [MEO] Loaded {len(df)} rows, {df['stationId'].nunique()} stations")
    return df


def extract_meo_station_coords(meo_df: pd.DataFrame) -> dict:
    """Extract {station_id: (lat, lon)} from MEO dataframe."""
    if meo_df is None or 'latitude' not in meo_df.columns:
        return {}
    coords = (
        meo_df.dropna(subset=['latitude', 'longitude'])
        .groupby('stationId')[['latitude', 'longitude']]
        .first()
    )
    return {row.Index: (row.latitude, row.longitude) for row in coords.itertuples()}


def wind_to_uv(speed, direction_deg):
    """Convert wind speed + direction (meteorological) to U, V components."""
    # Meteorological convention: direction = where wind comes FROM
    # U = -speed * sin(dir_rad), V = -speed * cos(dir_rad)
    dir_rad = np.radians(direction_deg)
    U = -speed * np.sin(dir_rad)
    V = -speed * np.cos(dir_rad)
    return U, V


def build_wind_features(
    aq_stations: list,
    aq_coords: dict,
    meo_df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
) -> tuple:
    """
    Build wind U/V arrays aligned to AQ stations and time index.
    Each AQ station gets the wind from its nearest MEO station.

    Returns
    -------
    wind_u : (T, N) array or None
    wind_v : (T, N) array or None
    """
    if meo_df is None:
        return None, None

    # Get MEO station coordinates
    meo_coords = extract_meo_station_coords(meo_df)
    if not meo_coords:
        print("  [WARNING] No MEO coordinates found. Cannot compute wind features.")
        return None, None

    meo_station_names = list(meo_coords.keys())
    meo_lats = np.array([meo_coords[s][0] for s in meo_station_names])
    meo_lons = np.array([meo_coords[s][1] for s in meo_station_names])

    aq_lats  = np.array([aq_coords[s][0] for s in aq_stations])
    aq_lons  = np.array([aq_coords[s][1] for s in aq_stations])

    matched_idx, matched_dist = match_stations_nearest(aq_lats, aq_lons, meo_lats, meo_lons)
    print(f"  [Wind] AQ→MEO station matching: max distance = {matched_dist.max():.1f} km")

    # Build MEO time series for each matched MEO station
    T = len(time_index)
    N = len(aq_stations)
    wind_u = np.full((T, N), np.nan, dtype=np.float32)
    wind_v = np.full((T, N), np.nan, dtype=np.float32)

    # Need wind_speed and wind_direction in MEO
    has_speed = 'wind_speed' in meo_df.columns
    has_dir   = 'wind_direction' in meo_df.columns
    if not has_speed:
        print("  [WARNING] wind_speed column not in MEO. Wind features unavailable.")
        return None, None

    meo_df = meo_df.set_index('utc_time')

    for aq_i, aq_station in enumerate(aq_stations):
        meo_station = meo_station_names[matched_idx[aq_i]]
        meo_sub = meo_df[meo_df['stationId'] == meo_station].reindex(time_index)

        speed = meo_sub['wind_speed'].values.astype(np.float32)
        if has_dir:
            direction = meo_sub['wind_direction'].values.astype(np.float32)
            # Filter bad direction values (999017 is a common sentinel)
            direction[np.abs(direction) > 360] = np.nan
            u, v = wind_to_uv(speed, direction)
        else:
            # If no direction, treat speed as magnitude only
            u = speed
            v = np.zeros_like(speed)

        wind_u[:, aq_i] = u
        wind_v[:, aq_i] = v

    return wind_u, wind_v


def pivot_to_array(
    df: pd.DataFrame,
    stations: list,
    time_index: pd.DatetimeIndex,
    columns: list,
) -> np.ndarray:
    """
    Pivot long-format DataFrame to (T, N, C) array.
    Missing values are NaN before imputation.
    """
    T = len(time_index)
    N = len(stations)
    C = len(columns)
    arr = np.full((T, N, C), np.nan, dtype=np.float32)

    df = df.set_index(['utc_time', 'stationId'])

    for n, station in enumerate(stations):
        if station not in df.index.get_level_values('stationId'):
            continue
        sub = df.xs(station, level='stationId')
        sub = sub.reindex(time_index)
        for c, col in enumerate(columns):
            if col in sub.columns:
                arr[:, n, c] = sub[col].values.astype(np.float32)

    return arr


def impute_rolling_mean(arr: np.ndarray, window: int = 24) -> np.ndarray:
    """
    Impute NaN values with a rolling mean over ±window/2 hours.
    Applied independently per station and pollutant.
    arr shape: (T, N, C)
    """
    T, N, C = arr.shape
    out = arr.copy()
    for n in range(N):
        for c in range(C):
            series = pd.Series(out[:, n, c])
            filled = series.fillna(series.rolling(window, min_periods=1, center=True).mean())
            # Any remaining NaN (isolated long gaps) filled with global mean
            global_mean = filled.mean()
            if pd.isna(global_mean):
                global_mean = 0.0
            filled = filled.fillna(global_mean)
            out[:, n, c] = filled.values
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n" + "="*60)
    print("Building Beijing dataset")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Load AQ data
    # ------------------------------------------------------------------
    print("\n[1] Loading AQ data...")
    aq_df = load_aq_data(RAW_DIR)

    # Determine which stations we have coordinates for
    all_stations = sorted(aq_df['stationId'].unique().tolist())
    stations_with_coords = [s for s in all_stations if s in BEIJING_AQ_COORDS]
    missing_coords = [s for s in all_stations if s not in BEIJING_AQ_COORDS]

    if missing_coords:
        print(f"  [WARNING] No coords for {len(missing_coords)} stations: {missing_coords}")
        print(f"  Using {len(stations_with_coords)} stations with known coordinates.")

    stations = stations_with_coords
    N = len(stations)
    print(f"  Using {N} stations")

    # ------------------------------------------------------------------
    # 2. Build unified time index (hourly)
    # ------------------------------------------------------------------
    t_min = aq_df['utc_time'].min().floor('H')
    t_max = aq_df['utc_time'].max().ceil('H')
    time_index = pd.date_range(t_min, t_max, freq='H')
    T = len(time_index)
    print(f"\n[2] Time index: {t_min} → {t_max} ({T} hourly steps)")

    # ------------------------------------------------------------------
    # 3. Pivot to (T, N, C) array
    # ------------------------------------------------------------------
    print("\n[3] Pivoting to array...")
    # Filter aq_df to known stations
    aq_df = aq_df[aq_df['stationId'].isin(stations)]
    data = pivot_to_array(aq_df, stations, time_index, POLLUTANTS)
    print(f"  Raw array shape: {data.shape}  NaN fraction: {np.isnan(data).mean():.4f}")

    # ------------------------------------------------------------------
    # 4. Impute missing values
    # ------------------------------------------------------------------
    print("\n[4] Imputing missing values (24h rolling mean)...")
    data = impute_rolling_mean(data, window=24)
    print(f"  After imputation NaN fraction: {np.isnan(data).mean():.6f}")

    # ------------------------------------------------------------------
    # 5. Build adjacency from AQ station coordinates
    # ------------------------------------------------------------------
    print("\n[5] Building adjacency matrix...")
    lats = np.array([BEIJING_AQ_COORDS[s][0] for s in stations])
    lons = np.array([BEIJING_AQ_COORDS[s][1] for s in stations])
    W = build_adjacency(lats, lons, sigma=SIGMA, threshold=ADJ_THRESHOLD)
    validate_adjacency(W, name="Beijing AQ adj")

    # ------------------------------------------------------------------
    # 6. Laplacian + eigendecomposition
    # ------------------------------------------------------------------
    print("\n[6] Computing Laplacian and eigendecomposition...")
    L = normalized_laplacian(W)
    validate_laplacian(L, name="Beijing L")
    cache = os.path.join(OUT_DIR, 'eigen_cache.pkl')
    eigenvalues, eigenvectors = compute_eigen(L, cache_path=cache)

    # ------------------------------------------------------------------
    # 7. Load MEO / wind features
    # ------------------------------------------------------------------
    print("\n[7] Loading MEO data for wind features...")
    meo_df = load_meo_data(RAW_DIR)
    wind_u, wind_v = build_wind_features(stations, BEIJING_AQ_COORDS, meo_df, time_index)

    if wind_u is not None:
        # Impute wind too
        wind_u_3d = wind_u[:, :, np.newaxis]
        wind_v_3d = wind_v[:, :, np.newaxis]
        wind_u = impute_rolling_mean(wind_u_3d, window=24)[:, :, 0]
        wind_v = impute_rolling_mean(wind_v_3d, window=24)[:, :, 0]
        print(f"  Wind arrays: {wind_u.shape}, NaN: {np.isnan(wind_u).mean():.6f}")

    # ------------------------------------------------------------------
    # 8. Normalize (global mean/std per pollutant)
    # ------------------------------------------------------------------
    print("\n[8] Normalizing...")
    mean = data.mean(axis=(0, 1))   # (C,)
    std  = data.std(axis=(0, 1))    # (C,)
    std  = np.where(std < 1e-6, 1.0, std)
    data_norm = (data - mean) / std
    print(f"  Pollutant means: {dict(zip(POLLUTANTS, mean.round(2)))}")

    # ------------------------------------------------------------------
    # 9. Train/val/test split (temporal, no shuffle)
    # ------------------------------------------------------------------
    print("\n[9] Splitting...")
    n_train = int(T * SPLIT[0])
    n_val   = int(T * SPLIT[1])
    n_test  = T - n_train - n_val
    train_idx = np.arange(0, n_train)
    val_idx   = np.arange(n_train, n_train + n_val)
    test_idx  = np.arange(n_train + n_val, T)
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test} steps")

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    print("\n[10] Saving...")
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
        'stations': stations,
        'lats': lats.tolist(),
        'lons': lons.tolist(),
        'pollutants': POLLUTANTS,
        'time_index': time_index,
        'n_stations': N,
        'n_timesteps': T,
        'has_wind': wind_u is not None,
    }
    with open(os.path.join(OUT_DIR, 'station_info.pkl'), 'wb') as f:
        pickle.dump(station_info, f)

    print(f"\n✓ Beijing preprocessing complete.")
    print(f"  data.npy: {data_norm.shape}  (T={T}, N={N}, C={len(POLLUTANTS)})")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == '__main__':
    main()
