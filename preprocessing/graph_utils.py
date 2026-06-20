"""
graph_utils.py
--------------
Shared graph construction utilities used by all dataset preprocessors.

Provides:
  - haversine distance computation
  - distance-based adjacency matrix (Gaussian kernel, thresholded)
  - normalized graph Laplacian
  - eigendecomposition with caching
  - validation utilities
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
import torch
import os
import pickle


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances (km) for arrays of lat/lon coordinates.

    Parameters
    ----------
    lats : (N,) array of latitudes in decimal degrees
    lons : (N,) array of longitudes in decimal degrees

    Returns
    -------
    D : (N, N) symmetric distance matrix in km
    """
    R = 6371.0  # Earth radius km
    coords = np.stack([np.radians(lats), np.radians(lons)], axis=1)  # (N, 2)

    N = len(lats)
    D = np.zeros((N, N))
    for i in range(N):
        dlat = coords[:, 0] - coords[i, 0]
        dlon = coords[:, 1] - coords[i, 1]
        a = np.sin(dlat / 2) ** 2 + (
            np.cos(coords[i, 0]) * np.cos(coords[:, 0]) * np.sin(dlon / 2) ** 2
        )
        D[i] = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return D


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------

def build_adjacency(
    lats: np.ndarray,
    lons: np.ndarray,
    sigma: float = None,
    threshold: float = 0.1,
    self_loops: bool = False,
) -> np.ndarray:
    """
    Build a symmetric weighted adjacency matrix using a Gaussian kernel on
    haversine distances, following the convention of AirPhyNet / DiffSTG.

        W_{ij} = exp(-d_{ij}^2 / sigma^2)   if d_{ij} < threshold_km_cutoff
               = 0                           otherwise

    Parameters
    ----------
    lats, lons : (N,) coordinate arrays
    sigma      : bandwidth in km. If None, set to the median pairwise distance.
    threshold  : edges with weight < threshold are zeroed (sparsification).
    self_loops : whether to include diagonal (default False).

    Returns
    -------
    W : (N, N) symmetric adjacency matrix, values in [0, 1].
    """
    D = haversine_matrix(lats, lons)
    N = D.shape[0]

    if sigma is None:
        # Use median non-zero distance as bandwidth
        upper = D[np.triu_indices(N, k=1)]
        sigma = float(np.median(upper))

    W = np.exp(-(D ** 2) / (sigma ** 2))
    W[W < threshold] = 0.0

    if not self_loops:
        np.fill_diagonal(W, 0.0)

    # Symmetrise (should already be symmetric, but enforce numerically)
    W = (W + W.T) / 2
    return W.astype(np.float32)


def build_adjacency_from_dist_matrix(
    D: np.ndarray,
    sigma: float = None,
    threshold: float = 0.1,
    self_loops: bool = False,
) -> np.ndarray:
    """
    Same as build_adjacency but accepts a precomputed distance matrix.
    Used for traffic datasets where adjacency is provided as road distances.
    """
    N = D.shape[0]
    if sigma is None:
        upper = D[np.triu_indices(N, k=1)]
        upper = upper[upper > 0]
        sigma = float(np.median(upper))

    W = np.exp(-(D ** 2) / (sigma ** 2))
    W[W < threshold] = 0.0

    if not self_loops:
        np.fill_diagonal(W, 0.0)

    W = (W + W.T) / 2
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------

def normalized_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric normalized Laplacian:
        L = I - D^{-1/2} W D^{-1/2}

    This is the standard form used in spectral GCN and the graph diffusion ODE.
    Eigenvalues lie in [0, 2].

    Parameters
    ----------
    W : (N, N) adjacency matrix (non-negative, symmetric)

    Returns
    -------
    L : (N, N) symmetric normalized Laplacian
    """
    N = W.shape[0]
    degree = W.sum(axis=1)  # (N,)

    # Handle isolated nodes (degree = 0)
    degree_inv_sqrt = np.where(degree > 0, degree ** -0.5, 0.0)
    D_inv_sqrt = np.diag(degree_inv_sqrt)

    L = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    # Enforce symmetry numerically
    L = (L + L.T) / 2
    return L.astype(np.float32)


def scaled_laplacian(W: np.ndarray) -> np.ndarray:
    """
    Scaled Laplacian: L_tilde = 2L/lambda_max - I
    Used in ChebNet; eigenvalues lie in [-1, 1].
    """
    L = normalized_laplacian(W)
    eigenvalues = np.linalg.eigvalsh(L)
    lambda_max = eigenvalues.max()
    if lambda_max < 1e-10:
        return L
    return (2.0 / lambda_max) * L - np.eye(L.shape[0])


# ---------------------------------------------------------------------------
# Eigendecomposition (core for SRG)
# ---------------------------------------------------------------------------

def compute_eigen(
    L: np.ndarray,
    cache_path: str = None,
) -> tuple:
    """
    Compute eigendecomposition of the symmetric normalized Laplacian:
        L = U Λ Uᵀ

    Uses scipy.linalg.eigh (exact, exploits symmetry, returns sorted eigenvalues).
    For N <= ~500, this is essentially instant. For larger N, takes a few seconds
    but is a one-time precomputation.

    Parameters
    ----------
    L          : (N, N) symmetric normalized Laplacian
    cache_path : if provided, save/load U and eigenvalues from disk

    Returns
    -------
    eigenvalues : (N,) array, sorted ascending, all >= 0
    eigenvectors: (N, N) array, columns are orthonormal eigenvectors
                  L = eigenvectors @ diag(eigenvalues) @ eigenvectors.T
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  [eigen] Loaded from cache: {cache_path}")
        return data['eigenvalues'], data['eigenvectors']

    print(f"  [eigen] Computing eigendecomposition for {L.shape[0]}x{L.shape[0]} Laplacian...")
    eigenvalues, eigenvectors = eigh(L)  # returns ascending order, real

    # Clamp small negative eigenvalues to 0 (numerical noise)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    print(f"  [eigen] Done. λ_min={eigenvalues.min():.6f}, λ_max={eigenvalues.max():.4f}")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}, f)
        print(f"  [eigen] Saved to cache: {cache_path}")

    return eigenvalues.astype(np.float32), eigenvectors.astype(np.float32)


def eigen_to_tensors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    device: str = 'cpu',
) -> tuple:
    """
    Convert numpy eigen arrays to PyTorch tensors for model use.

    Returns
    -------
    Lambda : (N,)   tensor of eigenvalues
    U      : (N, N) tensor of eigenvectors (columns)
    """
    Lambda = torch.tensor(eigenvalues, dtype=torch.float32, device=device)
    U = torch.tensor(eigenvectors, dtype=torch.float32, device=device)
    return Lambda, U


# ---------------------------------------------------------------------------
# Nearest-station matching (Beijing AQ ↔ MEO alignment)
# ---------------------------------------------------------------------------

def match_stations_nearest(
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
) -> np.ndarray:
    """
    For each target station, find the index of the nearest source station.
    Used to align Beijing AQ stations with MEO (meteorological) stations.

    Returns
    -------
    indices : (N_target,) int array, indices into source arrays
    distances: (N_target,) float array, matched distances in km
    """
    D = haversine_matrix(
        np.concatenate([target_lats, source_lats]),
        np.concatenate([target_lons, source_lons]),
    )
    N_t = len(target_lats)
    D_cross = D[:N_t, N_t:]  # (N_target, N_source)

    indices = D_cross.argmin(axis=1)
    distances = D_cross[np.arange(N_t), indices]
    return indices, distances


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_adjacency(W: np.ndarray, name: str = "W") -> None:
    """Sanity checks for adjacency matrix."""
    assert W.shape[0] == W.shape[1], f"{name}: not square"
    assert np.allclose(W, W.T, atol=1e-5), f"{name}: not symmetric"
    assert (W >= 0).all(), f"{name}: has negative values"
    assert (np.diag(W) == 0).all(), f"{name}: has self-loops"
    n_edges = (W > 0).sum() // 2
    density = n_edges / (W.shape[0] * (W.shape[0] - 1) / 2)
    print(f"  [{name}] N={W.shape[0]}, edges={n_edges}, density={density:.3f}, "
          f"w_max={W.max():.4f}, w_min={W[W>0].min():.6f}")


def validate_laplacian(L: np.ndarray, name: str = "L") -> None:
    """Sanity checks for Laplacian."""
    assert np.allclose(L, L.T, atol=1e-5), f"{name}: not symmetric"
    eigenvalues = np.linalg.eigvalsh(L)
    assert eigenvalues.min() >= -1e-5, f"{name}: has negative eigenvalues ({eigenvalues.min():.6f})"
    print(f"  [{name}] λ_min={eigenvalues.min():.6f}, λ_max={eigenvalues.max():.4f}, "
          f"rank={int((eigenvalues > 1e-6).sum())}/{L.shape[0]}")
