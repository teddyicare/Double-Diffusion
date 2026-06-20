"""Diagnose graph-prior k sensitivity and simple temporal priors.

This script is intentionally evaluation-only. It does not train models and does
not modify the cleaned project tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

from preprocessing.dataset import STDataset, get_dataset_dir, load_graph  # noqa: E402
from double_diffusion.seasonal import previous_period_indices  # noqa: E402


FIT_K = {
    "beijing": {"scalar": 0.1012, "per_channel": [0.0698, 0.0861, 0.0956, 0.1271, 0.1015, 0.1381]},
    "athens": {"scalar": 0.0645, "per_channel": [0.145, 0.131, 0.076, 0.021]},
    "pems08": {"scalar": 0.0201, "per_channel": [0.0064, 0.0226, 0.0289]},
    "pems04": {"scalar": 0.0248, "per_channel": [0.0098, 0.0343, 0.0297]},
}

PERIOD = {
    "beijing": 24,
    "athens": 24,
    "pems08": 288,
    "pems04": 288,
    "metr_la": 288,
    "pems_bay": 288,
}


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def normalized_mae(pred_raw: torch.Tensor, target_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> float:
    pred_norm = (pred_raw - mean.to(pred_raw.device)) / std.to(pred_raw.device)
    return float(torch.mean(torch.abs(pred_norm - target_norm)).item())


@torch.no_grad()
def graph_heat(
    x_last: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    k,
    n_steps: int,
    dt: float,
) -> torch.Tensor:
    """Scalar or per-channel graph heat prior."""
    bsz, n_nodes, n_channels = x_last.shape
    device = x_last.device
    lam = eigenvalues.to(device)
    u = eigenvectors.to(device)
    steps = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32)
    x0 = x_last.permute(1, 0, 2).reshape(n_nodes, bsz * n_channels)
    x0_spec = u.T @ x0
    if isinstance(k, (list, tuple)):
        k_vec = torch.tensor(k, device=device, dtype=torch.float32).view(1, 1, n_channels)
        filters = torch.exp(-k_vec * (lam.view(1, n_nodes, 1) * steps.view(n_steps, 1, 1) * dt))
    else:
        filters = torch.exp(-float(k) * lam.view(1, n_nodes) * steps.view(n_steps, 1) * dt)
    preds = []
    for h in range(n_steps):
        if filters.dim() == 3:
            spec = (x0_spec.reshape(n_nodes, bsz, n_channels) * filters[h].view(n_nodes, 1, n_channels)).reshape(
                n_nodes, bsz * n_channels
            )
        else:
            spec = x0_spec * filters[h].view(n_nodes, 1)
        xh = (u @ spec).reshape(n_nodes, bsz, n_channels).permute(1, 0, 2)
        preds.append(xh)
    return torch.stack(preds, dim=1)


def evaluate(dataset: str, future_len: int, k_values: list[float], dt_values: list[float], max_batches: int, device: str) -> dict:
    data_dir = get_dataset_dir(dataset, base_dir=str(CORE))
    ds = STDataset(data_dir, "test", history_len=12, future_len=future_len)
    graph = load_graph(data_dir, device=device)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    mean = ds.mean.float().to(device)
    std = ds.std.float().to(device)
    period = PERIOD.get(dataset, 24)
    fit = FIT_K.get(dataset, {})
    names = ["persist", "trend", "damped_trend", "seasonal"]
    for dt in dt_values:
        for k in k_values:
            names.append(f"ode_k{k:g}_dt{dt:g}")
            names.append(f"seasonal_graph_k{k:g}_dt{dt:g}")
            names.append(f"seasonal_graph_decay_k{k:g}_dt{dt:g}")
        if fit.get("scalar") is not None:
            names.append(f"ode_fit_scalar_dt{dt:g}")
            names.append(f"seasonal_graph_fit_scalar_dt{dt:g}")
        if fit.get("per_channel") is not None:
            names.append(f"ode_fit_perchan_dt{dt:g}")
            names.append(f"seasonal_graph_fit_perchan_dt{dt:g}")
    totals = {name: 0.0 for name in names}
    variations = {name: 0.0 for name in names if name.startswith("ode") or name.startswith("seasonal_graph")}
    n_obs = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        x_future = batch["x_future"].float().to(device)
        x_hist_raw = batch["x_hist_raw"].float().to(device)
        x_last_raw = x_hist_raw[:, -1]
        start_idx = batch["start_idx"].long()
        bsz = x_future.shape[0]

        persist = x_last_raw.unsqueeze(1).expand(-1, future_len, -1, -1)
        totals["persist"] += normalized_mae(persist, x_future, mean, std) * bsz

        hist_norm = (x_hist_raw - mean) / std
        r = min(6, hist_norm.shape[1] - 1)
        med_delta = (hist_norm[:, -r:] - hist_norm[:, -r - 1 : -1]).median(dim=1).values
        steps = torch.arange(1, future_len + 1, device=device).float().view(1, future_len, 1, 1)
        trend_norm = hist_norm[:, -1:].expand(-1, future_len, -1, -1) + steps * med_delta.unsqueeze(1)
        trend_raw = trend_norm * std + mean
        totals["trend"] += normalized_mae(trend_raw, x_future, mean, std) * bsz
        damp = 0.95 ** (steps - 1.0)
        damped_trend_norm = hist_norm[:, -1:].expand(-1, future_len, -1, -1) + damp * steps * med_delta.unsqueeze(1)
        damped_trend_raw = damped_trend_norm * std + mean
        totals["damped_trend"] += normalized_mae(damped_trend_raw, x_future, mean, std) * bsz

        seasonal_norm = []
        seasonal_now_raw = []
        for sample_idx, t in enumerate(start_idx.tolist()):
            cur = t - 1 - period
            # Leakage-free same-phase indices (strictly before t); see seasonal.py.
            idx = previous_period_indices(t, period, future_len)
            if idx is not None and cur >= 0:
                seasonal_norm.append(ds.data.index_select(0, torch.as_tensor(idx, dtype=torch.long)))
                seasonal_now_raw.append(ds.data_raw[cur])
            else:
                seasonal_norm.append(hist_norm[sample_idx, -1:].detach().cpu().expand(future_len, -1, -1))
                seasonal_now_raw.append(x_last_raw[sample_idx].detach().cpu())
        seasonal_norm_t = torch.stack(seasonal_norm, dim=0).to(device)
        seasonal_raw = seasonal_norm_t * std + mean
        seasonal_now_raw = torch.stack(seasonal_now_raw, dim=0).to(device)
        totals["seasonal"] += normalized_mae(seasonal_raw, x_future, mean, std) * bsz
        residual_now_raw = x_last_raw - seasonal_now_raw
        decay = (0.95 ** (torch.arange(1, future_len + 1, device=device).float() - 1.0)).view(1, future_len, 1, 1)

        for dt in dt_values:
            for k in k_values:
                name = f"ode_k{k:g}_dt{dt:g}"
                pred = graph_heat(x_last_raw, graph.eigenvalues, graph.eigenvectors, k, future_len, dt)
                totals[name] += normalized_mae(pred, x_future, mean, std) * bsz
                variations[name] += float((pred[:, -1] - pred[:, 0]).abs().mean().item()) * bsz
                res_pred = graph_heat(residual_now_raw, graph.eigenvalues, graph.eigenvectors, k, future_len, dt)
                sg_name = f"seasonal_graph_k{k:g}_dt{dt:g}"
                sg = seasonal_raw + res_pred
                totals[sg_name] += normalized_mae(sg, x_future, mean, std) * bsz
                variations[sg_name] += float((sg[:, -1] - sg[:, 0]).abs().mean().item()) * bsz
                sgd_name = f"seasonal_graph_decay_k{k:g}_dt{dt:g}"
                sgd = seasonal_raw + decay * res_pred
                totals[sgd_name] += normalized_mae(sgd, x_future, mean, std) * bsz
                variations[sgd_name] += float((sgd[:, -1] - sgd[:, 0]).abs().mean().item()) * bsz
            if fit.get("scalar") is not None:
                name = f"ode_fit_scalar_dt{dt:g}"
                pred = graph_heat(x_last_raw, graph.eigenvalues, graph.eigenvectors, fit["scalar"], future_len, dt)
                totals[name] += normalized_mae(pred, x_future, mean, std) * bsz
                variations[name] += float((pred[:, -1] - pred[:, 0]).abs().mean().item()) * bsz
                sg_name = f"seasonal_graph_fit_scalar_dt{dt:g}"
                sg = seasonal_raw + graph_heat(residual_now_raw, graph.eigenvalues, graph.eigenvectors, fit["scalar"], future_len, dt)
                totals[sg_name] += normalized_mae(sg, x_future, mean, std) * bsz
                variations[sg_name] += float((sg[:, -1] - sg[:, 0]).abs().mean().item()) * bsz
            if fit.get("per_channel") is not None:
                name = f"ode_fit_perchan_dt{dt:g}"
                pred = graph_heat(x_last_raw, graph.eigenvalues, graph.eigenvectors, fit["per_channel"], future_len, dt)
                totals[name] += normalized_mae(pred, x_future, mean, std) * bsz
                variations[name] += float((pred[:, -1] - pred[:, 0]).abs().mean().item()) * bsz
                sg_name = f"seasonal_graph_fit_perchan_dt{dt:g}"
                sg = seasonal_raw + graph_heat(residual_now_raw, graph.eigenvalues, graph.eigenvectors, fit["per_channel"], future_len, dt)
                totals[sg_name] += normalized_mae(sg, x_future, mean, std) * bsz
                variations[sg_name] += float((sg[:, -1] - sg[:, 0]).abs().mean().item()) * bsz
        n_obs += bsz

    metrics = {name: totals[name] / max(n_obs, 1) for name in names}
    var = {name: variations[name] / max(n_obs, 1) for name in variations}
    best = min(metrics, key=metrics.get)
    return {
        "dataset": dataset,
        "future_len": future_len,
        "n_samples": n_obs,
        "period": period,
        "fit_k": fit,
        "mae_norm": metrics,
        "ode_mean_abs_change_raw": var,
        "best": best,
        "best_mae_norm": metrics[best],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="beijing,pems08")
    parser.add_argument("--future-lens", default="12,24,48")
    parser.add_argument("--k-values", default="0,0.001,0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--dt-values", default="1")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="double_diffusion/outputs/prior_k_diagnostics.json")
    args = parser.parse_args()

    out = {}
    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        out[dataset] = {}
        for future_len in [int(x) for x in args.future_lens.split(",") if x.strip()]:
            print(f"diagnosing {dataset} F{future_len}")
            out[dataset][str(future_len)] = evaluate(
                dataset,
                future_len,
                parse_float_list(args.k_values),
                parse_float_list(args.dt_values),
                args.max_batches,
                args.device,
            )
            r = out[dataset][str(future_len)]
            print(f"  best={r['best']} mae={r['best_mae_norm']:.4f} persist={r['mae_norm']['persist']:.4f}")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
