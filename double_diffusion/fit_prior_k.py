"""Offline identification of the graph-diffusion coefficient k.

We select k by minimizing the NORMALIZED PRIOR RESIDUAL on a holdout split, with
NO neural training and NO gradient through the denoiser -- a cheap grid search
over the closed-form ODE prior.

Objective (default L2). The Resfusion denoiser is trained with a squared loss on a
target that carries the residual R(k) = (x_hat(k) - x0)/sigma, so its
schedule-weighted burden is E_s||w(s) R(k)||^2 = (schedule const) * ||R(k)||^2.
Hence the k that minimizes the L2 residual energy ||R(k)||^2 is provably the same
argmin as what training minimizes -- this is the coupling between the offline fit
and the in-training objective (see k_resfusion_link.md). L1 (MAE) is available via
--objective l1 but does NOT have this exact coupling (its argmin differs).

NOTE: ||R(k)|| is still a *proxy* for final MAE/CRPS (the denoiser may tolerate a
biased prior), so the chosen k* must also be checked against end metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

from preprocessing.dataset import STDataset, get_dataset_dir, load_graph  # noqa: E402
from double_diffusion.diagnose_prior_k import PERIOD, graph_heat, parse_float_list  # noqa: E402
from double_diffusion.seasonal import previous_period_indices  # noqa: E402


def norm(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (raw - mean.to(raw.device)) / std.to(raw.device)


def channel_residual(pred_raw: torch.Tensor, target_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, objective: str = "l2") -> torch.Tensor:
    """Per-channel normalized prior residual. objective='l2' -> mean squared
    residual energy ||R||^2 (the Resfusion-coupled objective); 'l1' -> mean |R|."""
    diff = norm(pred_raw, mean, std) - target_norm
    if objective == "l1":
        return torch.mean(torch.abs(diff), dim=(0, 1, 2))
    return torch.mean(diff * diff, dim=(0, 1, 2))


def seasonal_parts(ds: STDataset, batch: dict, future_len: int, period: int, device: str):
    x_hist_raw = batch["x_hist_raw"].float().to(device)
    x_last = x_hist_raw[:, -1]
    seasonal_norm = []
    seasonal_now_raw = []
    for sample_idx, start_value in enumerate(batch["start_idx"].long().tolist()):
        cur_idx = start_value - 1 - period
        # Leakage-free same-phase indices (strictly before start); see seasonal.py.
        idx = previous_period_indices(start_value, period, future_len)
        if idx is not None and cur_idx >= 0:
            seasonal_norm.append(ds.data.index_select(0, torch.as_tensor(idx, dtype=torch.long)))
            seasonal_now_raw.append(ds.data_raw[cur_idx])
        else:
            seasonal_norm.append(norm(x_last[sample_idx], ds.mean, ds.std).detach().cpu().unsqueeze(0).repeat(future_len, 1, 1))
            seasonal_now_raw.append(x_last[sample_idx].detach().cpu())
    return torch.stack(seasonal_norm, dim=0).to(device), torch.stack(seasonal_now_raw, dim=0).to(device)


def make_prior_raw(prior_mode, x_last_raw, seasonal_norm, seasonal_now_raw, mean, std, graph, k, future_len, dt, decay_rate):
    if prior_mode == "graph_ode":
        return graph_heat(x_last_raw, graph.eigenvalues, graph.eigenvectors, k, future_len, dt)
    seasonal_raw = seasonal_norm * std.to(seasonal_norm.device) + mean.to(seasonal_norm.device)
    residual = x_last_raw - seasonal_now_raw
    residual_pred = graph_heat(residual, graph.eigenvalues, graph.eigenvectors, k, future_len, dt)
    if prior_mode == "seasonal_graph_decay":
        decay = (decay_rate ** (torch.arange(1, future_len + 1, device=x_last_raw.device).float() - 1.0)).view(1, future_len, 1, 1)
        residual_pred = residual_pred * decay
    return seasonal_raw + residual_pred


@torch.no_grad()
def evaluate_k(dataset, future_len, prior_mode, k, dt, split, max_batches, device, decay_rate, objective):
    data_dir = get_dataset_dir(dataset, base_dir=str(CORE))
    ds = STDataset(data_dir, split, history_len=12, future_len=future_len)
    graph = load_graph(data_dir, device=device)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    mean = ds.mean.float().to(device)
    std = ds.std.float().to(device)
    period = PERIOD.get(dataset, 24)
    total = torch.zeros(ds.n_features, device=device)
    n = 0
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x_future = batch["x_future"].float().to(device)
        x_last_raw = batch["x_hist_raw"].float().to(device)[:, -1]
        seasonal_norm, seasonal_now_raw = seasonal_parts(ds, batch, future_len, period, device)
        pred = make_prior_raw(prior_mode, x_last_raw, seasonal_norm, seasonal_now_raw, mean, std, graph, k, future_len, dt, decay_rate)
        err = channel_residual(pred, x_future, mean, std, objective)
        total += err * x_future.shape[0]
        n += x_future.shape[0]
    per_channel = total / max(n, 1)
    return {
        "score": float(per_channel.mean().item()),
        "per_channel_score": [float(x) for x in per_channel.detach().cpu()],
        "n": n,
    }


def fit_one(dataset, future_len, prior_mode, k_values, dt, split, max_batches, device, decay_rate, objective):
    scalar_rows = []
    for k in k_values:
        row = evaluate_k(dataset, future_len, prior_mode, k, dt, split, max_batches, device, decay_rate, objective)
        row["k"] = k
        scalar_rows.append(row)
    best_scalar = min(scalar_rows, key=lambda x: x["score"])

    n_channels = len(best_scalar["per_channel_score"])
    best_k = []
    for ch in range(n_channels):
        row = min(scalar_rows, key=lambda x: x["per_channel_score"][ch])
        best_k.append(row["k"])
    per = evaluate_k(dataset, future_len, prior_mode, best_k, dt, split, max_batches, device, decay_rate, objective)
    return {
        "objective": objective,
        "scalar": {"k": best_scalar["k"], "score": best_scalar["score"], "per_channel_score": best_scalar["per_channel_score"]},
        "per_channel": {"k": best_k, "score": per["score"], "per_channel_score": per["per_channel_score"]},
        "grid": scalar_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="beijing,pems08")
    parser.add_argument("--future-lens", default="24,48")
    parser.add_argument("--prior-modes", default="graph_ode,seasonal_graph,seasonal_graph_decay")
    parser.add_argument("--k-values", default="0,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.15,0.2")
    parser.add_argument("--objective", default="l2", choices=["l2", "l1"], help="l2 = ||R||^2 residual energy (Resfusion-coupled); l1 = MAE.")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--decay-rate", type=float, default=0.95)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="double_diffusion/outputs/kfit_auto.json")
    args = parser.parse_args()

    out = {}
    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        out[dataset] = {}
        for future_len in [int(x) for x in args.future_lens.split(",") if x.strip()]:
            out[dataset][str(future_len)] = {}
            for prior_mode in [x.strip() for x in args.prior_modes.split(",") if x.strip()]:
                print(f"fit[{args.objective}] {dataset} F{future_len} {prior_mode}")
                out[dataset][str(future_len)][prior_mode] = fit_one(
                    dataset, future_len, prior_mode,
                    parse_float_list(args.k_values), args.dt, args.split,
                    args.max_batches, args.device, args.decay_rate, args.objective,
                )
                r = out[dataset][str(future_len)][prior_mode]
                print(f"  scalar k={r['scalar']['k']} score={r['scalar']['score']:.5f} | per-channel k={r['per_channel']['k']}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
