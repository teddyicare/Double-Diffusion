"""Spatial-gate diagnostic.

For each dataset it computes, on the TRAIN split, the prior residual (normalized) at the paper's
horizon H'=24 with the FITTED k*, projects it onto the graph-Laplacian eigenbasis, and reports the
fraction of residual spectral energy in the low graph-frequency band at the mid horizon step. A high
fraction means the residual is spatially smooth (the low-pass prior already captured the recoverable
structure -> gate OFF); a low fraction means it carries high graph-frequency structure (gate ON).

Band = lower half of the graph spectrum by eigenvalue (lambda <= median); a lowest-third variant is
reported for robustness. The TEMPORAL control projects the same residual onto the temporal axis
instead of the graph and should NOT separate the domains.

Run from the repo root after the datasets are built under ``data/<name>/processed``:
    python diagnostics/run_spatial_gate_diagnostic.py
It writes ``diagnostics/spectral_gate_output.json`` (read-only with respect to the data).
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing.dataset import get_dataset_dir, get_dataloaders   # noqa: E402
from double_diffusion.models import graph_heat_forecast              # noqa: E402

KFIT_PATH = ROOT / "configs" / "kfit_l2_hpc.json"
assert KFIT_PATH.exists(), f"kfit_l2_hpc.json not found at {KFIT_PATH}"
KFIT = json.load(open(KFIT_PATH))


def kstar(ds: str) -> float:
    # the scalar fit is what the DD runs use; grid-min by l2 score reproduces it
    return float(KFIT[ds]["24"]["graph_ode"]["scalar"]["k"])


H, HP = 12, 24
DATASETS = ["beijing", "athens", "pems08", "pems04"]


def main() -> None:
    out = {}
    for ds in DATASETS:
        dd = get_dataset_dir(ds, base_dir=str(ROOT))
        train_loader, _, _, graph = get_dataloaders(
            data_dir=dd, history_len=H, future_len=HP, batch_size=64, num_workers=0, device="cpu")
        mean = torch.tensor(np.load(os.path.join(dd, "mean.npy")), dtype=torch.float32)
        std = torch.tensor(np.load(os.path.join(dd, "std.npy")), dtype=torch.float32)
        U = graph.eigenvectors.float().cpu()        # (N,N), columns = eigenvectors
        lam = graph.eigenvalues.float().cpu()        # (N,)
        N = int(graph.n_nodes)
        k = kstar(ds)

        Fp = HP // 2 + 1
        graph_energy = torch.zeros(HP, N)            # residual spectral energy (graph axis), summed
        temporal_energy = torch.zeros(Fp)            # CONTROL: residual energy by temporal frequency
        nwin = 0
        with torch.no_grad():
            for batch in train_loader:
                x_last = batch["x_hist_raw"][:, -1]                              # (B,N,C) raw
                x_future = batch["x_future"]                                     # (B,HP,N,C) normalized
                prior_raw = graph_heat_forecast(x_last, lam, U, k, HP, dt=1.0)   # (B,HP,N,C) raw
                prior = (prior_raw - mean.view(1, 1, 1, -1)) / std.view(1, 1, 1, -1)
                res = prior - x_future                                           # (B,HP,N,C) normalized residual
                spec = torch.einsum("in,bhic->bhnc", U, res)                     # project node->graph mode
                graph_energy += (spec ** 2).sum(dim=(0, 3))                      # (HP,N)
                tfreq = torch.fft.rfft(res, dim=1)                               # (B,Fp,N,C) temporal spectrum
                temporal_energy += (tfreq.abs() ** 2).sum(dim=(0, 2, 3))         # (Fp,)
                nwin += res.shape[0]

        order = torch.argsort(lam)
        low_half = order[: N // 2]
        low_third = order[: max(1, N // 3)]
        h_mid = HP // 2

        def gratio(modes, h):
            tot = graph_energy[h].sum()
            return float((graph_energy[h, modes].sum() / tot).clamp(0, 1))

        tlow = temporal_energy[: Fp // 2].sum() / temporal_energy.sum()
        out[ds] = {
            "k_star": k, "history": H, "horizon": HP, "n_nodes": N, "n_windows": int(nwin),
            "low_band_def": "lambda <= median (lower half of graph spectrum)",
            "low_band_ratio_h_mid": gratio(low_half, h_mid),
            "low_band_ratio_h_1": gratio(low_half, 0),
            "low_band_ratio_h_end": gratio(low_half, HP - 1),
            "low_third_ratio_h_mid": gratio(low_third, h_mid),
            "temporal_low_freq_ratio": float(tlow),   # control axis (should NOT separate domains)
        }
        # AUTOMATIC GATE RULE (deterministic, no forecast evaluation):
        # rho_G = fraction of prior-residual graph-spectral energy in the lowest-third band.
        # rho_G <  1/2 -> substantial high graph-frequency structure remains -> gate ON.
        # rho_G >= 1/2 -> residual already smooth, prior captured it           -> gate OFF.
        rho_G = out[ds]["low_third_ratio_h_mid"]
        out[ds]["rho_G"] = rho_G
        out[ds]["gate"] = "on" if rho_G < 0.5 else "off"
        print(f"{ds:8s} k*={k} N={N} Hp={HP}: rho_G(low-third)={rho_G:.3f} -> gate {out[ds]['gate'].upper():3s} "
              f"| graph-low(median)={out[ds]['low_band_ratio_h_mid']:.3f} "
              f"temporal-low(control)={out[ds]['temporal_low_freq_ratio']:.3f}", flush=True)

    dst = ROOT / "diagnostics" / "spectral_gate_output.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
