"""Train and evaluate the industry-track validation variants.

Example pilot:
  python -B double_diffusion/run_validation.py --dataset beijing --future-len 24 \
      --epochs 2 --max-train-batches 40 --max-val-batches 8 --max-test-batches 8 \
      --d-model 24 --n-blocks 2 --diffusion-steps 24 --n-samples 4 \
      --rolling-train --reveal-choices 0,1,3,6 --eval-policies initial,rolling,no_update,reweight
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CORE))

from preprocessing.dataset import get_dataset_dir, get_dataloaders  # noqa: E402
from prior.physics_ode import PhysicsODESolver, normalize_with_stats  # noqa: E402
from double_diffusion.models import ResidualDiffusionCalibrator  # noqa: E402
from double_diffusion.calibration import fit_conformal_widening  # noqa: E402


def parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_list(value: str) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std.to(x.device) + mean.to(x.device)


def suffix_mask(target: torch.Tensor, reveal_prefix: int) -> torch.Tensor:
    mask = torch.ones_like(target, dtype=torch.bool)
    if reveal_prefix > 0:
        mask[:, :reveal_prefix] = False
    return mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask].mean() if mask.any() else x.mean()


def crps_sorted(samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    # Empirical CRPS in normalized units.
    bsz, n_samples, future_len, n_nodes, n_channels = samples.shape
    term1 = torch.abs(samples - target.unsqueeze(1)).mean(dim=1)
    sorted_samples, _ = torch.sort(samples, dim=1)
    idx = torch.arange(1, n_samples + 1, device=samples.device, dtype=samples.dtype)
    coeff = (2 * idx - n_samples - 1).reshape(1, n_samples, 1, 1, 1)
    term2 = (coeff * sorted_samples).sum(dim=1) / float(n_samples * n_samples)
    values = term1 - term2
    return masked_mean(values, mask).item()


def interval_stats(samples: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, mask: torch.Tensor, conformal_q: torch.Tensor | None = None) -> dict:
    samples_raw = denormalize(samples, mean, std)
    target_raw = denormalize(target, mean, std)
    q05 = torch.quantile(samples_raw, 0.05, dim=1)
    q95 = torch.quantile(samples_raw, 0.95, dim=1)
    coverage = ((target_raw >= q05) & (target_raw <= q95)).float()
    width = q95 - q05
    out = {
        "p90_coverage": masked_mean(coverage, mask).item(),
        "p90_width": masked_mean(width, mask).item(),
    }
    if conformal_q is not None:
        # Additive split-conformal widening, fit on validation (H', C) -> broadcast.
        q = conformal_q.to(samples_raw.device).view(1, conformal_q.shape[0], 1, conformal_q.shape[1])
        lo = q05 - q
        hi = q95 + q
        # Guard against pathological over-shrink (Q very negative): keep hi >= lo.
        mid = 0.5 * (lo + hi)
        lo = torch.minimum(lo, mid)
        hi = torch.maximum(hi, mid)
        cov_cal = ((target_raw >= lo) & (target_raw <= hi)).float()
        out["p90_coverage_cal"] = masked_mean(cov_cal, mask).item()
        out["p90_width_cal"] = masked_mean(hi - lo, mask).item()
    return out


def point_metrics(samples: torch.Tensor, target: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, reveal_prefix: int, conformal_q: torch.Tensor | None = None) -> dict:
    mask = suffix_mask(target, reveal_prefix)
    pred_raw = denormalize(samples.mean(dim=1), mean, std)
    target_raw = denormalize(target, mean, std)
    err = pred_raw - target_raw
    out = {
        "mae": masked_mean(torch.abs(err), mask).item(),
        "rmse": torch.sqrt(masked_mean(err.pow(2), mask)).item(),
        "crps": crps_sorted(samples, target, mask),
    }
    out.update(interval_stats(samples, target, mean, std, mask, conformal_q=conformal_q))
    return out


@torch.no_grad()
def graph_ode_samples(batch: dict, graph, mean: torch.Tensor, std: torch.Tensor, future_len: int, k: float, n_samples: int, prior_mode: str) -> torch.Tensor:
    device = graph.eigenvalues.device
    x_hist_raw = batch["x_hist_raw"].to(device)
    x_last = x_hist_raw[:, -1]
    if prior_mode == "persistence":
        pred_raw = x_last.unsqueeze(1).repeat(1, future_len, 1, 1)
    else:
        ode = PhysicsODESolver(graph.eigenvalues, graph.eigenvectors, k=k).to(device)
        pred_raw = ode(x_last, n_steps=future_len)
    pred = normalize_with_stats(pred_raw, mean.to(device), std.to(device))
    return pred.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)


def reweight_samples(samples: torch.Tensor, target: torch.Tensor, reveal_prefix: int, n_out: int, temperature: float) -> torch.Tensor:
    if reveal_prefix <= 0:
        return samples[:, :n_out]
    prefix_err = (samples[:, :, :reveal_prefix] - target[:, None, :reveal_prefix]).pow(2).mean(dim=(2, 3, 4))
    if temperature <= 0:
        scale = torch.median(prefix_err.detach()).clamp_min(1e-4)
    else:
        scale = torch.tensor(temperature, device=samples.device, dtype=samples.dtype)
    weights = torch.softmax(-prefix_err / scale, dim=1)
    idx = torch.multinomial(weights, num_samples=n_out, replacement=True)
    gather_idx = idx.reshape(idx.shape[0], n_out, 1, 1, 1).expand(-1, -1, samples.shape[2], samples.shape[3], samples.shape[4])
    return torch.gather(samples, dim=1, index=gather_idx)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def evaluate_policy(
    model,
    loader,
    graph,
    mean: torch.Tensor,
    std: torch.Tensor,
    args,
    policy: str,
    reveal_prefix: int,
    max_batches: int,
    conformal_q: torch.Tensor | None = None,
) -> dict:
    device = args.device
    totals: dict[str, float] = {}
    count = 0
    wall = 0.0
    batches_done = 0
    # Region the metrics are averaged over. 'initial'/'graph_ode' are scored over
    # the full horizon [0, future_len); 'rolling'/'no_update'/'reweight' are scored
    # only over the unrevealed suffix [reveal_prefix, future_len). NEVER compare a
    # full-horizon initial against a suffix-only rolling: the fair operational
    # comparison is rolling_rN vs no_update_rN (both scored over the same suffix N).
    score_prefix = 0 if policy in ("initial", "graph_ode") else reveal_prefix
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        target = batch["x_future"].to(device)
        cuda_sync()
        start = time.perf_counter()
        if policy == "graph_ode":
            if model is not None:
                prior = model.make_prior(batch, mean.to(device), std.to(device))
                samples = prior.unsqueeze(1).repeat(1, args.n_samples, 1, 1, 1)
            else:
                samples = graph_ode_samples(batch, graph, mean, std, args.future_len, args.k, args.n_samples, args.prior_mode)
        elif policy == "initial":
            samples = model.predict(batch, mean, std, n_samples=args.n_samples, reveal_prefix=0)
        elif policy == "rolling":
            samples = model.predict(
                batch,
                mean,
                std,
                n_samples=args.n_samples,
                reveal_prefix=reveal_prefix,
                start_step=args.update_steps,
            )
        elif policy == "no_update":
            samples = model.predict(batch, mean, std, n_samples=args.n_samples, reveal_prefix=0)
        elif policy == "reweight":
            pool = max(args.reweight_pool, args.n_samples)
            pool_samples = model.predict(batch, mean, std, n_samples=pool, reveal_prefix=0)
            samples = reweight_samples(pool_samples, target, reveal_prefix, args.n_samples, args.reweight_temperature)
        else:
            raise ValueError(f"Unknown policy: {policy}")
        cuda_sync()
        wall += time.perf_counter() - start
        metrics = point_metrics(samples, target, mean.to(device), std.to(device), score_prefix, conformal_q=conformal_q)
        batch_size = target.shape[0]
        count += batch_size
        batches_done += 1
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    out = {key: value / max(count, 1) for key, value in totals.items()}
    out["seconds"] = wall
    out["batches"] = batches_done
    out["samples"] = count
    out["seconds_per_batch"] = wall / max(out["batches"], 1)
    # Self-document the scored horizon so tables cannot silently compare
    # metrics computed over different step ranges (see score_prefix above).
    out["score_prefix"] = score_prefix
    out["score_steps"] = f"{score_prefix}-{args.future_len - 1}"
    return out


def train_loss(model, loader, mean: torch.Tensor, std: torch.Tensor, args, train: bool, optimizer=None, scaler=None) -> float:
    model.train(train)
    total = 0.0
    batches = 0
    max_batches = args.max_train_batches if train else args.max_val_batches
    reveal_choices = parse_int_list(args.reveal_choices) if args.rolling_train else [0]
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            if args.amp and args.device.startswith("cuda"):
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model.training_step(batch, mean, std, reveal_choices=reveal_choices)
            else:
                loss = model.training_step(batch, mean, std, reveal_choices=reveal_choices)
        if train:
            if scaler is not None and args.amp and args.device.startswith("cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        total += float(loss.detach().cpu().item())
        batches += 1
    return total / max(batches, 1)


def load_data(args):
    data_dir = get_dataset_dir(args.dataset, base_dir=str(CORE))
    return get_dataloaders(
        data_dir=data_dir,
        history_len=args.history_len,
        future_len=args.future_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


def build_model(args, graph, n_channels: int) -> ResidualDiffusionCalibrator:
    k_value = parse_float_list(args.k_per_channel) or args.k
    return ResidualDiffusionCalibrator(
        graph=graph,
        n_channels=n_channels,
        history_len=args.history_len,
        future_len=args.future_len,
        d_model=args.d_model,
        n_blocks=args.n_blocks,
        k=k_value,
        diffusion_steps=args.diffusion_steps,
        beta_end=args.beta_end,
        schedule=args.schedule,
        n_samples=args.n_samples,
        temporal_mode=args.temporal_mode,
        use_channel=not args.no_channel,
        use_spatial=not args.no_spatial,
        use_coarse=args.coarse_branch,
        cheb_order=args.cheb_order,
        dropout=args.dropout,
        prior_mode=args.prior_mode,
        dt=args.dt,
        prior_data_norm=args.prior_data_norm,
        prior_data_raw=args.prior_data_raw,
        prior_period=args.prior_period,
        residual_decay=args.residual_decay,
    ).to(args.device)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["beijing", "athens", "pems04", "pems08", "metr_la", "pems_bay"])
    parser.add_argument("--model", default="fastdd", choices=["fastdd", "graph_ode"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--future-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--temporal-mode", default="fft", choices=["fft", "conv", "none"])
    parser.add_argument("--coarse-branch", action="store_true")
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument("--no-spatial", action="store_true")
    parser.add_argument("--cheb-order", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--prior-mode",
        default="graph_ode",
        choices=[
            "graph_ode",
            "persistence",
            "zero",
            "seasonal",
            "damped_trend",
            "seasonal_graph",
            "seasonal_graph_residual",
            "seasonal_graph_decay",
        ],
    )
    parser.add_argument("--k", type=float, default=0.1)
    parser.add_argument("--k-per-channel", default=None, help="Comma-separated per-channel graph diffusion k.")
    parser.add_argument("--auto-k-file", default=None, help="Load k from fit_prior_k.py output.")
    parser.add_argument("--auto-k-kind", default="scalar", choices=["scalar", "per_channel"])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--prior-period", type=int, default=None)
    parser.add_argument("--residual-decay", type=float, default=0.95)

    parser.add_argument("--diffusion-steps", type=int, default=48)
    parser.add_argument("--beta-end", type=float, default=0.2)
    parser.add_argument("--schedule", default="linear", choices=["linear", "quadratic", "cosine"])
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--rolling-train", action="store_true")
    parser.add_argument("--reveal-choices", default="0,1,3,6,12")
    parser.add_argument("--eval-policies", default="graph_ode,initial,rolling,no_update,reweight")
    parser.add_argument("--reveal-prefixes", default="3,6,12")
    parser.add_argument("--update-steps", type=int, default=8)
    parser.add_argument("--reweight-pool", type=int, default=12)
    parser.add_argument("--reweight-temperature", type=float, default=0.0)
    parser.add_argument("--calibrate", default="none", choices=["none", "conformal"], help="Post-hoc interval calibration fit on the validation split.")
    parser.add_argument("--calib-alpha", type=float, default=0.10, help="Miscoverage level; 0.10 -> 90%% intervals.")
    parser.add_argument("--max-calib-batches", type=int, default=0, help="Cap val batches used to fit conformal (0 = use max-val-batches).")

    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    parser.add_argument("--output-root", default=str(ROOT / "double_diffusion" / "outputs"))
    parser.add_argument("--eval-only", default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    train_loader, val_loader, test_loader, graph = load_data(args)
    train_ds = train_loader.dataset
    mean = train_ds.mean.float().to(args.device)
    std = train_ds.std.float().to(args.device)
    n_channels = train_ds.n_features
    if args.prior_period is None:
        args.prior_period = 288 if args.dataset in ("pems04", "pems08", "metr_la", "pems_bay") else 24
    args.prior_data_norm = train_ds.data
    args.prior_data_raw = train_ds.data_raw
    if args.auto_k_file:
        with open(args.auto_k_file, "r", encoding="utf-8") as f:
            kfit = json.load(f)
        selected = kfit.get(args.dataset, {}).get(str(args.future_len), {}).get(args.prior_mode)
        if selected is None:
            selected = kfit.get(str(args.future_len), {}).get(args.prior_mode) or kfit.get(args.prior_mode)
        if selected is None:
            raise RuntimeError(f"No k entry for dataset={args.dataset}, future={args.future_len}, prior={args.prior_mode}")
        if args.auto_k_kind == "per_channel":
            args.k_per_channel = ",".join(str(x) for x in selected["per_channel"]["k"])
        else:
            args.k = float(selected["scalar"]["k"])
    run_name = args.run_name or (
        f"{args.dataset}_f{args.future_len}_{args.model}_d{args.d_model}_b{args.n_blocks}_"
        f"{args.temporal_mode}_coarse{int(args.coarse_branch)}_roll{int(args.rolling_train)}_s{args.seed}"
    )
    out_dir = Path(args.output_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args_for_json = {
        key: value
        for key, value in vars(args).items()
        if key not in ("prior_data_norm", "prior_data_raw")
    }
    payload = {
        "args": args_for_json,
        "run_name": run_name,
        "s_prime": None,
        "train": [],
        "eval": {},
    }

    model = None
    if args.model == "fastdd":
        model = build_model(args, graph, n_channels)
        payload["s_prime"] = model.s_prime
        n_params = sum(p.numel() for p in model.parameters())
        payload["params"] = n_params
        checkpoint = out_dir / "best_model.pt"
        if args.eval_only:
            model.load_state_dict(torch.load(args.eval_only, map_location=args.device))
        else:
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))
            best_val = math.inf
            bad = 0
            for epoch in range(1, args.epochs + 1):
                start = time.perf_counter()
                tr = train_loss(model, train_loader, mean, std, args, train=True, optimizer=optimizer, scaler=scaler)
                val = train_loss(model, val_loader, mean, std, args, train=False)
                elapsed = time.perf_counter() - start
                row = {"epoch": epoch, "train_loss": tr, "val_loss": val, "seconds": elapsed}
                payload["train"].append(row)
                save_json(out_dir / "results.json", payload)
                print(f"epoch {epoch:03d} train_loss={tr:.5f} val_loss={val:.5f} seconds={elapsed:.1f}")
                if val < best_val:
                    best_val = val
                    bad = 0
                    torch.save(model.state_dict(), checkpoint)
                else:
                    bad += 1
                if bad >= args.patience:
                    print(f"early stop at epoch {epoch}")
                    break
            if checkpoint.exists():
                model.load_state_dict(torch.load(checkpoint, map_location=args.device))

    policies = parse_str_list(args.eval_policies)
    if args.model == "graph_ode":
        policies = ["graph_ode"]
    prefixes = parse_int_list(args.reveal_prefixes)

    # Post-hoc conformal interval calibration, fit on the validation split ONLY
    # (no test leakage). It is policy-specific: the prediction configuration
    # (reveal prefix, reverse start step) changes the interval scale, so we fit
    # one widening per configuration and apply the matching one at test.
    if args.calibrate == "conformal" and model is not None:
        model.eval()
        payload["conformal_alpha"] = args.calib_alpha
    conformal_cache: dict = {}

    def conformal_for(reveal_prefix: int, start_step) -> torch.Tensor | None:
        if args.calibrate != "conformal" or model is None:
            return None
        cache_key = (reveal_prefix, start_step)
        if cache_key not in conformal_cache:
            conformal_cache[cache_key] = fit_conformal_widening(
                model,
                val_loader,
                mean,
                std,
                n_samples=args.n_samples,
                alpha=args.calib_alpha,
                reveal_prefix=reveal_prefix,
                start_step=start_step,
                max_batches=args.max_calib_batches or args.max_val_batches,
                device=args.device,
            )
        return conformal_cache[cache_key]

    for policy in policies:
        if policy in ("graph_ode", "initial"):
            prefix_list = [0]
        else:
            prefix_list = [p for p in prefixes if 0 <= p < args.future_len]
        for prefix in prefix_list:
            key = f"{policy}_r{prefix}"
            # Match the conformal widening to this policy's prediction config.
            if policy == "rolling":
                cq = conformal_for(prefix, args.update_steps)
            elif policy in ("initial", "no_update"):
                cq = conformal_for(0, None)
            else:  # graph_ode (deterministic) / reweight (pool) -> no interval calibration
                cq = None
            print(f"evaluating {key}")
            metrics = evaluate_policy(
                model=model,
                loader=test_loader,
                graph=graph,
                mean=mean,
                std=std,
                args=args,
                policy=policy,
                reveal_prefix=prefix,
                max_batches=args.max_test_batches,
                conformal_q=cq,
            )
            payload["eval"][key] = metrics
            save_json(out_dir / "results.json", payload)

    save_json(out_dir / "results.json", payload)
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
