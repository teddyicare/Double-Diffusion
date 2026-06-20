"""
dataset.py
----------
Unified PyTorch Dataset for all four datasets (Beijing, Athens, METR-LA, PEMS-BAY).

Provides:
  - STDataset: sliding window dataset returning (history, future, physics_pred, graph)
  - get_dataloader: convenience function returning train/val/test DataLoaders
  - GraphData: named tuple holding graph tensors (adj, L, eigenvalues, eigenvectors)
"""

import numpy as np
import os
import pickle
from typing import Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Graph data container
# ---------------------------------------------------------------------------

@dataclass
class GraphData:
    """All graph-related tensors, preloaded once and shared across batches."""
    adj:          torch.Tensor   # (N, N)
    laplacian:    torch.Tensor   # (N, N) normalized L
    eigenvalues:  torch.Tensor   # (N,)   sorted ascending
    eigenvectors: torch.Tensor   # (N, N) columns are eigenvectors
    n_nodes:      int

    def to(self, device):
        return GraphData(
            adj=self.adj.to(device),
            laplacian=self.laplacian.to(device),
            eigenvalues=self.eigenvalues.to(device),
            eigenvectors=self.eigenvectors.to(device),
            n_nodes=self.n_nodes,
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class STDataset(Dataset):
    """
    Sliding-window spatio-temporal dataset.

    Each sample contains:
      x_hist   : (H, N, C)  normalized historical readings
      x_future : (H', N, C) normalized future ground truth
      wind_hist: (H, N, 2)  wind U/V for history (zeros if unavailable)
      wind_fut : (H', N, 2) wind U/V for future window (for AirPhyNet advection)
      indices  : scalar int, start index (for debugging)

    The physics ODE prediction is computed on-the-fly during training
    by the model (not precomputed here), to allow flexibility in k.
    """

    def __init__(
        self,
        data_dir:    str,
        split:       str,           # 'train', 'val', or 'test'
        history_len: int = 12,
        future_len:  int = 12,
    ):
        self.history_len = history_len
        self.future_len  = future_len
        self.window      = history_len + future_len

        # Load normalized data: (T, N, C)
        data_path = os.path.join(data_dir, 'data.npy')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Processed data not found: {data_path}\n"
                f"Run the appropriate preprocessing script first."
            )
        self.data = torch.tensor(
            np.load(data_path), dtype=torch.float32
        )  # (T, N, C)

        # Load split indices
        splits = np.load(os.path.join(data_dir, 'splits.npz'))
        idx = splits[split]

        # Valid start indices: need history_len steps before + future_len after
        valid_start = idx[idx >= history_len]
        valid_start = valid_start[valid_start + future_len <= len(self.data)]
        self.indices = valid_start

        # Load wind features (optional)
        wind_u_path = os.path.join(data_dir, 'wind_u.npy')
        wind_v_path = os.path.join(data_dir, 'wind_v.npy')
        if os.path.exists(wind_u_path) and os.path.exists(wind_v_path):
            wind_u = np.load(wind_u_path)  # (T, N)
            wind_v = np.load(wind_v_path)  # (T, N)
            self.wind = torch.tensor(
                np.stack([wind_u, wind_v], axis=-1), dtype=torch.float32
            )  # (T, N, 2)
            self.has_wind = True
        else:
            self.has_wind = False
            N = self.data.shape[1]
            T = self.data.shape[0]
            self.wind = torch.zeros(T, N, 2, dtype=torch.float32)

        # Load normalization stats
        self.mean = torch.tensor(np.load(os.path.join(data_dir, 'mean.npy')))
        self.std  = torch.tensor(np.load(os.path.join(data_dir, 'std.npy')))

        # Raw (unnormalized) data for physics ODE initialization.
        # Always reconstruct from imputed normalized data to guarantee no NaN.
        # (data_raw.npy may contain NaN for audit purposes, e.g., METR-LA sensors.)
        self.data_raw = self.data * self.std + self.mean

        # Station info
        info_path = os.path.join(data_dir, 'station_info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as f:
                self.station_info = pickle.load(f)
        else:
            self.station_info = {}

        T, N, C = self.data.shape
        print(f"  STDataset [{split}]: {len(self.indices)} samples | "
              f"T={T}, N={N}, C={C} | wind={self.has_wind}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        t = int(self.indices[idx])
        H  = self.history_len
        Hp = self.future_len

        # Normalized data
        x_hist   = self.data[t - H : t]         # (H,  N, C)
        x_future = self.data[t     : t + Hp]    # (H', N, C)

        # Raw data for ODE initialization (unnormalized)
        x_hist_raw   = self.data_raw[t - H : t]
        x_future_raw = self.data_raw[t     : t + Hp]

        # Wind
        wind_hist = self.wind[t - H : t]         # (H,  N, 2)
        wind_fut  = self.wind[t     : t + Hp]    # (H', N, 2)

        return {
            'x_hist':        x_hist,        # (H,  N, C) normalized
            'x_future':      x_future,      # (H', N, C) normalized
            'x_hist_raw':    x_hist_raw,    # (H,  N, C) raw, for ODE init
            'x_future_raw':  x_future_raw,  # (H', N, C) raw, for residual
            'wind_hist':     wind_hist,     # (H,  N, 2)
            'wind_fut':      wind_fut,      # (H', N, 2)
            'start_idx':     t,
        }

    @property
    def n_nodes(self) -> int:
        return self.data.shape[1]

    @property
    def n_features(self) -> int:
        return self.data.shape[2]


# ---------------------------------------------------------------------------
# Graph loader (separate from dataset — loaded once, not per-sample)
# ---------------------------------------------------------------------------

def load_graph(data_dir: str, device: str = 'cpu') -> GraphData:
    """
    Load precomputed graph tensors from processed data directory.
    These are shared across all splits and loaded once.
    """
    adj  = torch.tensor(np.load(os.path.join(data_dir, 'adj.npy')),
                        dtype=torch.float32)
    L    = torch.tensor(np.load(os.path.join(data_dir, 'laplacian.npy')),
                        dtype=torch.float32)
    lam  = torch.tensor(np.load(os.path.join(data_dir, 'eigenvalues.npy')),
                        dtype=torch.float32)
    U    = torch.tensor(np.load(os.path.join(data_dir, 'eigenvectors.npy')),
                        dtype=torch.float32)

    graph = GraphData(
        adj=adj, laplacian=L, eigenvalues=lam, eigenvectors=U,
        n_nodes=adj.shape[0],
    )
    return graph.to(device)


# ---------------------------------------------------------------------------
# Convenience DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir:      str,
    history_len:   int = 12,
    future_len:    int = 12,
    batch_size:    int = 32,
    num_workers:   int = 4,
    device:        str = 'cpu',
) -> Tuple[DataLoader, DataLoader, DataLoader, GraphData]:
    """
    Build train / val / test DataLoaders and load graph tensors.

    Returns
    -------
    train_loader, val_loader, test_loader, graph
    """
    graph = load_graph(data_dir, device=device)

    def make_loader(split, shuffle, batch):
        ds = STDataset(data_dir, split, history_len, future_len)
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device != 'cpu'),
            drop_last=(split == 'train'),
        )

    train_loader = make_loader('train', shuffle=True,  batch=batch_size)
    val_loader   = make_loader('val',   shuffle=False, batch=batch_size * 2)
    test_loader  = make_loader('test',  shuffle=False, batch=batch_size * 2)

    return train_loader, val_loader, test_loader, graph


# ---------------------------------------------------------------------------
# Dataset registry — maps string name → processed data directory
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'beijing':  'data/beijing/processed',
    'athens':   'data/athens/processed',
    'metr_la':  'data/metr_la/processed',
    'pems_bay': 'data/pems_bay/processed',
    'pems04':   'data/pems04/processed',
    'pems08':   'data/pems08/processed',
}


def get_dataset_dir(name: str, base_dir: str = None) -> str:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    rel = DATASET_REGISTRY[name]
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), '..')
    return os.path.join(base_dir, rel)
