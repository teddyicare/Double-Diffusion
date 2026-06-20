"""Fast residual-diffusion models for the SIGSPATIAL industry pilot.

The code in this file is deliberately self-contained. It imports only the
existing graph ODE helper from the cleaned project tree and keeps the new
denoiser/rolling-calibration logic isolated in this validation workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from double_diffusion.seasonal import previous_period_indices


def sinusoidal_embedding(steps: torch.Tensor, dim: int) -> torch.Tensor:
    """Diffusion-step sinusoidal embedding."""
    device = steps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = steps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


@torch.no_grad()
def graph_heat_forecast(
    x_last: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    k,
    n_steps: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Scalar or per-channel graph heat forecast in raw units."""
    bsz, n_nodes, n_channels = x_last.shape
    device = x_last.device
    lam = eigenvalues.to(device)
    u = eigenvectors.to(device)
    steps = torch.arange(1, n_steps + 1, device=device, dtype=torch.float32)

    x0 = x_last.permute(1, 0, 2).reshape(n_nodes, bsz * n_channels)
    x0_spec = u.T @ x0
    if isinstance(k, (list, tuple)) or torch.is_tensor(k):
        k_vec = torch.as_tensor(k, dtype=torch.float32, device=device).flatten()
        if k_vec.numel() != n_channels:
            raise ValueError(f"per-channel k has {k_vec.numel()} values, expected {n_channels}")
        filters = torch.exp(
            -k_vec.view(1, 1, n_channels)
            * lam.view(1, n_nodes, 1)
            * steps.view(n_steps, 1, 1)
            * dt
        )
    else:
        filters = torch.exp(-float(k) * lam.view(1, n_nodes) * steps.view(n_steps, 1) * dt)

    preds = []
    for step_idx in range(n_steps):
        if filters.dim() == 3:
            spec = (
                x0_spec.reshape(n_nodes, bsz, n_channels)
                * filters[step_idx].view(n_nodes, 1, n_channels)
            ).reshape(n_nodes, bsz * n_channels)
        else:
            spec = x0_spec * filters[step_idx].view(n_nodes, 1)
        pred = (u @ spec).reshape(n_nodes, bsz, n_channels).permute(1, 0, 2)
        preds.append(pred)
    return torch.stack(preds, dim=1)


class SpectralGMLPTemporal(nn.Module):
    """Attention-free temporal mixing with an rFFT filter and gated MLP."""

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1
        self.filter_real = nn.Parameter(torch.randn(self.n_freq, d_model) * 0.02)
        self.filter_imag = nn.Parameter(torch.randn(self.n_freq, d_model) * 0.02)
        self.proj_in = nn.Linear(d_model, 2 * d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        # cuFFT rejects non-power-of-two sizes in half precision, so this rFFT
        # block must run in float32 even under AMP autocast (seq_len = H + H' is
        # typically not a power of two, e.g. 36 or 60).
        with torch.autocast(device_type=h.device.type, enabled=False):
            h = h.float()
            hf = torch.fft.rfft(h, n=self.seq_len, dim=1)
            filt = torch.complex(self.filter_real, self.filter_imag)
            hf = hf * filt.unsqueeze(0)
            mag = torch.abs(hf)
            gate, value = self.proj_in(mag).chunk(2, dim=-1)
            scale = self.proj_out(torch.sigmoid(gate) * value)
            out = torch.fft.irfft(hf * (1.0 + scale), n=self.seq_len, dim=1)
            out = self.norm(residual.float() + out)
        return out.to(residual.dtype)


class DilatedTemporalConv(nn.Module):
    """Fast local temporal mixer; useful as a speed/performance ablation."""

    def __init__(self, seq_len: int, d_model: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=d_model,
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=1),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        x = self.norm(h).transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x).transpose(1, 2)
        return residual + x


class ChannelMixer(nn.Module):
    """MLP-Mixer style channel gate over pollutants or traffic variables."""

    def __init__(self, n_channels: int, d_model: int):
        super().__init__()
        hidden_c = max(2 * n_channels, 4)
        self.token_in = nn.Linear(n_channels, hidden_c)
        self.token_out = nn.Linear(hidden_c, n_channels)
        self.feature_in = nn.Linear(d_model, 2 * d_model)
        self.feature_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h
        x = self.norm1(h).transpose(1, 2)
        x = self.token_out(F.gelu(self.token_in(x))).transpose(1, 2)
        h = residual + x
        residual = h
        gate, value = self.feature_in(self.norm2(h)).chunk(2, dim=-1)
        return residual + self.feature_out(torch.sigmoid(gate) * value)


class ChebGraphGate(nn.Module):
    """Chebyshev graph convolution gate, O(K|E|D) with sparse Laplacian matmul."""

    def __init__(
        self,
        laplacian: torch.Tensor,
        lambda_max: float,
        n_channels: int,
        d_model: int,
        order: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_nodes = laplacian.shape[0]
        eye = torch.eye(n_nodes, device=laplacian.device, dtype=laplacian.dtype)
        l_scaled = (2.0 / max(float(lambda_max), 1e-6)) * laplacian - eye
        # Keep the scaled Laplacian DENSE. Our graphs are small (N<=~300), where a dense GEMM
        # is vastly faster than torch.sparse.mm: sparse mm has heavy per-call overhead and is
        # invoked per-Chebyshev-order x per-block x per-reverse-step, which made the spatial
        # gate ~9x the lean cost (161 vs 18 s/batch on PEMS08). Dense is trivial here (307^2 ~ 0.4MB).
        self.register_buffer("l_scaled", l_scaled.float())
        self.order = order
        self.weights = nn.Parameter(torch.randn(order + 1, n_channels, d_model, d_model) * 0.01)
        with torch.no_grad():
            for c_idx in range(n_channels):
                self.weights[0, c_idx].copy_(torch.eye(d_model) * 0.5)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, C, D)
        bsz, time_len, n_nodes, n_channels, d_model = h.shape
        residual = h
        flat = h.permute(2, 0, 1, 3, 4).reshape(n_nodes, bsz * time_len * n_channels * d_model)
        t0 = flat
        t1 = self.l_scaled @ flat

        def to_btcn(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(n_nodes, bsz * time_len, n_channels, d_model).permute(1, 2, 0, 3)

        out = torch.einsum("bcnd,cde->bcne", to_btcn(t0), self.weights[0])
        out = out + torch.einsum("bcnd,cde->bcne", to_btcn(t1), self.weights[1])
        prev, curr = t0, t1
        for order in range(2, self.order + 1):
            nxt = 2 * (self.l_scaled @ curr) - prev
            out = out + torch.einsum("bcnd,cde->bcne", to_btcn(nxt), self.weights[order])
            prev, curr = curr, nxt
        out = out.reshape(bsz, time_len, n_channels, n_nodes, d_model).permute(0, 1, 3, 2, 4)
        out = self.dropout(out)
        return self.norm((residual + out).reshape(-1, d_model)).reshape_as(h)


class CoarseTemporalBranch(nn.Module):
    """A single U-Net-like coarse temporal branch: downsample, mix, upsample, fuse."""

    def __init__(self, seq_len: int, d_model: int, mode: str = "fft", dropout: float = 0.1):
        super().__init__()
        coarse_len = max(2, math.ceil(seq_len / 2))
        if mode == "conv":
            self.mixer = DilatedTemporalConv(coarse_len, d_model, dilation=1, dropout=dropout)
        else:
            self.mixer = SpectralGMLPTemporal(coarse_len, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bsz, time_len, n_nodes, n_channels, d_model = h.shape
        x = h.permute(0, 2, 3, 4, 1).reshape(bsz * n_nodes * n_channels, d_model, time_len)
        coarse = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.mixer(coarse.transpose(1, 2)).transpose(1, 2)
        up = F.interpolate(coarse, size=time_len, mode="linear", align_corners=False)
        up = up.reshape(bsz, n_nodes, n_channels, d_model, time_len).permute(0, 4, 1, 2, 3)
        return h + torch.tanh(self.gate) * self.proj(up)


class FastTTGBlock(nn.Module):
    """Temporal, channel, and graph gates with CSDI-style residual/skip output."""

    def __init__(
        self,
        seq_len: int,
        n_nodes: int,
        n_channels: int,
        d_model: int,
        laplacian: torch.Tensor,
        lambda_max: float,
        temporal_mode: str = "fft",
        use_channel: bool = True,
        use_spatial: bool = True,
        use_coarse: bool = False,
        cheb_order: int = 3,
        dropout: float = 0.1,
        block_index: int = 0,
    ):
        super().__init__()
        if temporal_mode == "conv":
            dilation = 1 + (block_index % 3)
            self.temporal = DilatedTemporalConv(seq_len, d_model, dilation=dilation, dropout=dropout)
        elif temporal_mode == "none":
            self.temporal = nn.Identity()
        else:
            self.temporal = SpectralGMLPTemporal(seq_len, d_model)
        self.channel = ChannelMixer(n_channels, d_model) if use_channel else nn.Identity()
        self.spatial = (
            ChebGraphGate(laplacian, lambda_max, n_channels, d_model, order=cheb_order, dropout=dropout)
            if use_spatial
            else nn.Identity()
        )
        self.coarse = (
            CoarseTemporalBranch(seq_len, d_model, mode="conv" if temporal_mode == "conv" else "fft", dropout=dropout)
            if use_coarse
            else nn.Identity()
        )
        self.step_proj = nn.Linear(d_model, d_model)
        self.mid = nn.Linear(d_model, 2 * d_model)
        self.cond = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, 2 * d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, side: torch.Tensor, step_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, time_len, n_nodes, n_channels, d_model = h.shape
        h = h + self.step_proj(step_emb).reshape(bsz, 1, 1, 1, d_model)

        x = h.permute(0, 2, 3, 1, 4).reshape(bsz * n_nodes * n_channels, time_len, d_model)
        x = self.temporal(x)
        h = x.reshape(bsz, n_nodes, n_channels, time_len, d_model).permute(0, 3, 1, 2, 4)

        x = h.reshape(bsz * time_len * n_nodes, n_channels, d_model)
        x = self.channel(x)
        h = x.reshape(bsz, time_len, n_nodes, n_channels, d_model)

        h = self.spatial(h)
        h = self.coarse(h)

        gate, value = (self.mid(h) + self.cond(side)).chunk(2, dim=-1)
        h_out = self.drop(torch.sigmoid(gate) * torch.tanh(value))
        residual, skip = self.out_proj(h_out).chunk(2, dim=-1)
        return residual, skip


class FastTTGDenoiser(nn.Module):
    """Lightweight TTG denoiser with an optional single coarse temporal branch."""

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        n_blocks: int,
        seq_len: int,
        n_nodes: int,
        laplacian: torch.Tensor,
        eigenvalues: torch.Tensor,
        temporal_mode: str = "fft",
        use_channel: bool = True,
        use_spatial: bool = True,
        use_coarse: bool = False,
        cheb_order: int = 3,
        dropout: float = 0.1,
        max_steps: int = 101,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.input_proj = nn.Linear(3, d_model)
        self.step_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.node_emb = nn.Embedding(n_nodes, d_model)
        self.chan_emb = nn.Embedding(n_channels, d_model)
        self.register_buffer("time_pos", self._time_pos_table(seq_len, d_model))
        self.side_proj = nn.Linear(3 * d_model + 2, d_model)
        lambda_max = float(eigenvalues.max().item())
        self.blocks = nn.ModuleList(
            [
                FastTTGBlock(
                    seq_len=seq_len,
                    n_nodes=n_nodes,
                    n_channels=n_channels,
                    d_model=d_model,
                    laplacian=laplacian,
                    lambda_max=lambda_max,
                    temporal_mode=temporal_mode,
                    use_channel=use_channel,
                    use_spatial=use_spatial,
                    use_coarse=use_coarse,
                    cheb_order=cheb_order,
                    dropout=dropout,
                    block_index=i,
                )
                for i in range(n_blocks)
            ]
        )
        self.skip_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))

    @staticmethod
    def _time_pos_table(seq_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(seq_len).float().unsqueeze(1)
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / max(half - 1, 1))
        args = pos * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def _side_info(self, obs_mask: torch.Tensor) -> torch.Tensor:
        bsz, time_len, n_nodes, n_channels = obs_mask.shape
        device = obs_mask.device
        tp = self.time_pos[:time_len].to(device).reshape(1, time_len, 1, 1, -1).expand(
            bsz, time_len, n_nodes, n_channels, -1
        )
        nid = self.node_emb(torch.arange(n_nodes, device=device)).reshape(1, 1, n_nodes, 1, -1).expand(
            bsz, time_len, n_nodes, n_channels, -1
        )
        cid = self.chan_emb(torch.arange(n_channels, device=device)).reshape(1, 1, 1, n_channels, -1).expand(
            bsz, time_len, n_nodes, n_channels, -1
        )
        lead = torch.linspace(0, 1, time_len, device=device).reshape(1, time_len, 1, 1, 1).expand(
            bsz, time_len, n_nodes, n_channels, 1
        )
        side = torch.cat([tp, nid, cid, obs_mask.unsqueeze(-1), lead], dim=-1)
        return self.side_proj(side)

    def forward(self, x_s: torch.Tensor, x_cond: torch.Tensor, obs_mask: torch.Tensor, s: torch.Tensor,
                side: torch.Tensor | None = None) -> torch.Tensor:
        stacked = torch.stack([x_s, x_cond, obs_mask], dim=-1)
        h = self.input_proj(stacked)
        step_emb = self.step_mlp(sinusoidal_embedding(s, self.d_model))
        # side info depends only on obs_mask (constant across the reverse chain), so the caller
        # can compute it once and pass it in -- avoids a ~(B*K*T*N*C)x(3d+2) matmul + multi-GB
        # allocation on every one of the S' reverse steps.
        if side is None:
            side = self._side_info(obs_mask)
        skip_sum: torch.Tensor | float = 0.0
        for block in self.blocks:
            residual, skip = block(h, side, step_emb)
            h = (h + residual) / math.sqrt(2.0)
            skip_sum = skip_sum + skip
        skip_sum = self.skip_norm(skip_sum / math.sqrt(len(self.blocks)))
        return self.head(skip_sum).squeeze(-1)


@dataclass
class Context:
    x0_full: torch.Tensor
    x_cond: torch.Tensor
    obs_mask: torch.Tensor
    res_full: torch.Tensor
    reveal_prefix: int


class ResidualDiffusionCalibrator(nn.Module):
    """Graph-prior residual diffusion with masked rolling calibration support."""

    def __init__(
        self,
        graph,
        n_channels: int,
        history_len: int,
        future_len: int,
        d_model: int = 32,
        n_blocks: int = 4,
        k: float = 0.1,
        diffusion_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.2,
        schedule: str = "linear",
        n_samples: int = 8,
        temporal_mode: str = "fft",
        use_channel: bool = True,
        use_spatial: bool = True,
        use_coarse: bool = False,
        cheb_order: int = 3,
        dropout: float = 0.1,
        prior_mode: str = "graph_ode",
        dt: float = 1.0,
        prior_data_norm: Optional[torch.Tensor] = None,
        prior_data_raw: Optional[torch.Tensor] = None,
        prior_period: int = 24,
        residual_decay: float = 0.95,
    ):
        super().__init__()
        self.history_len = history_len
        self.future_len = future_len
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.S = diffusion_steps
        self.prior_mode = prior_mode
        self.k = k
        self.dt = dt
        self.prior_period = int(prior_period)
        self.residual_decay = float(residual_decay)
        self.prior_data_norm = prior_data_norm.float().cpu() if prior_data_norm is not None else None
        self.prior_data_raw = prior_data_raw.float().cpu() if prior_data_raw is not None else None
        self.register_buffer("graph_eigenvalues", graph.eigenvalues.float())
        self.register_buffer("graph_eigenvectors", graph.eigenvectors.float())
        self.denoiser = FastTTGDenoiser(
            n_channels=n_channels,
            d_model=d_model,
            n_blocks=n_blocks,
            seq_len=history_len + future_len,
            n_nodes=graph.n_nodes,
            laplacian=graph.laplacian,
            eigenvalues=graph.eigenvalues,
            temporal_mode=temporal_mode,
            use_channel=use_channel,
            use_spatial=use_spatial,
            use_coarse=use_coarse,
            cheb_order=cheb_order,
            dropout=dropout,
            max_steps=diffusion_steps + 1,
        )
        # Noise schedule shape. "linear" is the default; "quadratic" packs more small
        # betas early (gentler initial corruption), "cosine" is the Nichol-Dhariwal alpha-bar
        # schedule. s_prime is recomputed from the actual schedule below, so the accelerated
        # reverse start (and hence latency) adapts automatically to the shape.
        if schedule == "quadratic":
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, diffusion_steps) ** 2
        elif schedule == "cosine":
            steps = diffusion_steps
            t = torch.linspace(0, steps, steps + 1) / steps
            f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            ab = f / f[0]
            betas = (1.0 - ab[1:] / ab[:-1]).clamp(min=1e-5, max=0.999)
        else:  # linear
            betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_1_ab", torch.sqrt(1 - alpha_bar))
        self.s_prime = int(torch.argmin(torch.abs(self.sqrt_ab - 0.5)).item()) + 1

    def _seasonal_raw(self, batch: dict, mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = mean.device
        x_hist_raw = batch["x_hist_raw"].to(device)
        bsz, _, n_nodes, n_channels = x_hist_raw.shape
        x_last = x_hist_raw[:, -1]
        if self.prior_data_raw is None:
            fallback = x_last.unsqueeze(1).repeat(1, self.future_len, 1, 1)
            return fallback, x_last, torch.zeros(bsz, device=device, dtype=torch.bool)

        data_raw = self.prior_data_raw
        seasonal_fut = []
        seasonal_now = []
        valid = []
        for sample_pos, start_value in enumerate(batch["start_idx"].detach().cpu().long().tolist()):
            cur_idx = start_value - 1 - self.prior_period
            # Leakage-free same-phase lookup: every index is strictly before the
            # forecast origin even when future_len > prior_period (see seasonal.py).
            idx = previous_period_indices(start_value, self.prior_period, self.future_len)
            if idx is not None and cur_idx >= 0:
                seasonal_fut.append(data_raw.index_select(0, torch.as_tensor(idx, dtype=torch.long)))
                seasonal_now.append(data_raw[cur_idx])
                valid.append(True)
            else:
                seasonal_fut.append(x_last[sample_pos].detach().cpu().unsqueeze(0).repeat(self.future_len, 1, 1))
                seasonal_now.append(x_last[sample_pos].detach().cpu())
                valid.append(False)
        fut = torch.stack(seasonal_fut, dim=0).to(device)
        now = torch.stack(seasonal_now, dim=0).to(device)
        valid_t = torch.tensor(valid, device=device, dtype=torch.bool)
        return fut, now, valid_t

    def _damped_trend_norm(self, batch: dict) -> torch.Tensor:
        x_hist = batch["x_hist"].to(next(self.parameters()).device)
        hist_len = x_hist.shape[1]
        recent = min(6, hist_len - 1)
        deltas = x_hist[:, -recent:] - x_hist[:, -recent - 1 : -1]
        med_delta = deltas.median(dim=1).values
        steps = torch.arange(1, self.future_len + 1, device=x_hist.device, dtype=x_hist.dtype).view(1, self.future_len, 1, 1)
        decay = self.residual_decay ** (steps - 1.0)
        return x_hist[:, -1:].repeat(1, self.future_len, 1, 1) + decay * steps * med_delta.unsqueeze(1)

    def make_prior(self, batch: dict, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        from prior.physics_ode import normalize_with_stats

        device = next(self.parameters()).device
        mean = mean.to(device)
        std = std.to(device)
        x_hist_raw = batch["x_hist_raw"].to(device)
        x_last = x_hist_raw[:, -1]
        if self.prior_mode == "persistence":
            pred_raw = x_last.unsqueeze(1).repeat(1, self.future_len, 1, 1)
            return normalize_with_stats(pred_raw, mean, std)
        elif self.prior_mode == "zero":
            pred_raw = mean.reshape(1, 1, 1, -1).repeat(
                x_hist_raw.shape[0], self.future_len, x_hist_raw.shape[2], 1
            )
            return normalize_with_stats(pred_raw, mean, std)
        elif self.prior_mode == "seasonal":
            seasonal_fut, _, _ = self._seasonal_raw(batch, mean, std)
            return normalize_with_stats(seasonal_fut, mean, std)
        elif self.prior_mode == "damped_trend":
            return self._damped_trend_norm(batch)
        elif self.prior_mode in ("seasonal_graph", "seasonal_graph_residual", "seasonal_graph_decay"):
            seasonal_fut, seasonal_now, _ = self._seasonal_raw(batch, mean, std)
            residual_now = x_last - seasonal_now
            residual_pred = graph_heat_forecast(
                residual_now,
                self.graph_eigenvalues,
                self.graph_eigenvectors,
                self.k,
                self.future_len,
                dt=self.dt,
            )
            if self.prior_mode == "seasonal_graph_decay":
                steps = torch.arange(1, self.future_len + 1, device=device, dtype=residual_pred.dtype).view(1, self.future_len, 1, 1)
                residual_pred = residual_pred * (self.residual_decay ** (steps - 1.0))
            pred_raw = seasonal_fut + residual_pred
            return normalize_with_stats(pred_raw, mean, std)
        else:
            pred_raw = graph_heat_forecast(
                x_last,
                self.graph_eigenvalues,
                self.graph_eigenvectors,
                self.k,
                self.future_len,
                dt=self.dt,
            )
            return normalize_with_stats(pred_raw, mean, std)

    def build_context(
        self,
        batch: dict,
        mean: torch.Tensor,
        std: torch.Tensor,
        reveal_prefix: int = 0,
    ) -> Context:
        device = next(self.parameters()).device
        x_hist = batch["x_hist"].to(device)
        x_future = batch["x_future"].to(device)
        x_hist_raw = batch["x_hist_raw"].to(device)
        bsz, hist_len, n_nodes, n_channels = x_hist.shape
        reveal_prefix = int(max(0, min(reveal_prefix, self.future_len - 1)))
        prior = self.make_prior(batch, mean.to(device), std.to(device))
        x0_full = torch.cat([x_hist, x_future], dim=1)
        x_cond = torch.cat([x_hist, prior], dim=1)
        obs_mask = torch.zeros_like(x0_full)
        obs_mask[:, :hist_len] = 1.0
        if reveal_prefix > 0:
            x_cond[:, hist_len : hist_len + reveal_prefix] = x_future[:, :reveal_prefix]
            obs_mask[:, hist_len : hist_len + reveal_prefix] = 1.0
        res_future = prior - x_future
        if reveal_prefix > 0:
            res_future[:, :reveal_prefix] = 0.0
        res_full = torch.cat([torch.zeros(bsz, hist_len, n_nodes, n_channels, device=device), res_future], dim=1)
        return Context(x0_full=x0_full, x_cond=x_cond, obs_mask=obs_mask, res_full=res_full, reveal_prefix=reveal_prefix)

    def q_sample(self, x0: torch.Tensor, res: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_idx = s - 1
        sqrt_ab = self.sqrt_ab[s_idx].reshape(-1, 1, 1, 1)
        sqrt_1_ab = self.sqrt_1_ab[s_idx].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        x_s = sqrt_ab * x0 + (1 - sqrt_ab) * res + sqrt_1_ab * eps
        beta_s = self.betas[s_idx].reshape(-1, 1, 1, 1)
        alpha_s = self.alphas[s_idx].reshape(-1, 1, 1, 1)
        eps_tilde = eps + ((1 - torch.sqrt(alpha_s)) * sqrt_1_ab / beta_s) * res
        return x_s, eps_tilde

    def _draw_reveal(self, choices: Iterable[int], device: torch.device) -> int:
        valid = [int(c) for c in choices if 0 <= int(c) < self.future_len]
        if not valid:
            return 0
        idx = torch.randint(0, len(valid), (1,), device=device).item()
        return valid[idx]

    def training_step(
        self,
        batch: dict,
        mean: torch.Tensor,
        std: torch.Tensor,
        reveal_choices: Iterable[int] = (0,),
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        reveal_prefix = self._draw_reveal(reveal_choices, device)
        ctx = self.build_context(batch, mean, std, reveal_prefix=reveal_prefix)
        bsz = ctx.x0_full.shape[0]
        s = torch.randint(1, self.s_prime + 1, (bsz,), device=device)
        x_s, eps_tilde = self.q_sample(ctx.x0_full, ctx.res_full, s)
        x_s = x_s * (1.0 - ctx.obs_mask) + ctx.x0_full * ctx.obs_mask
        eps_hat = self.denoiser(x_s, ctx.x_cond, ctx.obs_mask, s)
        future_obs = ctx.obs_mask[:, self.history_len :]
        train_mask = 1.0 - future_obs
        loss = ((eps_hat[:, self.history_len :] - eps_tilde[:, self.history_len :]) ** 2 * train_mask).sum()
        return loss / train_mask.sum().clamp_min(1.0)

    @torch.no_grad()
    def predict(
        self,
        batch: dict,
        mean: torch.Tensor,
        std: torch.Tensor,
        n_samples: Optional[int] = None,
        reveal_prefix: int = 0,
        start_step: Optional[int] = None,
    ) -> torch.Tensor:
        if n_samples is None:
            n_samples = self.n_samples
        device = next(self.parameters()).device
        ctx = self.build_context(batch, mean, std, reveal_prefix=reveal_prefix)
        bsz = ctx.x0_full.shape[0]
        start = int(start_step or self.s_prime)
        start = max(1, min(start, self.s_prime))
        sqrt_ab = self.sqrt_ab[start - 1]
        sqrt_1_ab = self.sqrt_1_ab[start - 1]
        # Vectorize the ensemble. The previous code ran the full reverse chain once PER sample
        # (n_samples sequential chains = n_samples*start tiny forwards on batch bsz), which
        # underutilizes the GPU and re-runs the graph gate on every sample. Instead, replicate
        # the conditioning along the sample dim and run the chain ONCE over a (bsz*chunk) batch:
        # 'start' forwards per chunk on a large batch (vs DiffSTG's full S steps). Identical
        # Monte-Carlo draws, ~chunk-fold fewer kernel launches. Chunk to bound activation memory.
        # Memory-safe ensemble batching. The 5D (B,T,N,C,d) denoiser has a large activation
        # footprint that grows with the graph size N (many permute/reshape copies + FFT
        # intermediates), so cap the effective batch by node count: roughly constant peak memory
        # across datasets. Falls back to chunk=1 (== per-sample, the original memory profile) on
        # big graphs so it never OOMs, while still batching the ensemble when there is headroom.
        n_nodes_ctx = ctx.x0_full.shape[2]
        # Match the original memory profile on big graphs: chunk=1 (== batch B per forward, which
        # ran fine) when bsz*N is already large; only batch the ensemble on small graphs where
        # there is headroom. 20000/N -> traffic (N>=170) stays chunk=1; air quality batches.
        max_bk = max(bsz, int(20000 / max(n_nodes_ctx, 1)))
        chunk = max(1, min(n_samples, max_bk // max(bsz, 1)))
        groups = []
        remaining = n_samples
        while remaining > 0:
            kc = min(chunk, remaining)
            x_cond = ctx.x_cond.repeat_interleave(kc, dim=0)
            obs_mask = ctx.obs_mask.repeat_interleave(kc, dim=0)
            x0_full = ctx.x0_full.repeat_interleave(kc, dim=0)
            bk = bsz * kc
            x_s = sqrt_ab * x_cond + sqrt_1_ab * torch.randn_like(x_cond)
            x_s = x_s * (1.0 - obs_mask) + x0_full * obs_mask
            side = self.denoiser._side_info(obs_mask)  # constant across the chain -> compute once
            for s_idx in range(start, 0, -1):
                s_tensor = torch.full((bk,), s_idx, dtype=torch.long, device=device)
                # The denoiser forward is ~45% dense Linear + ~13% LayerNorm; in fp32 that never
                # touches the tensor cores. Run it in bf16 (native on H100, no range issues) for
                # ~2x. The rFFT block self-disables autocast internally, so it stays fp32 as cuFFT
                # requires. mu/var update below stays fp32 for stability.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    eps_hat = self.denoiser(x_s, x_cond, obs_mask, s_tensor, side=side)
                eps_hat = eps_hat.float()
                beta_s = self.betas[s_idx - 1]
                alpha_s = self.alphas[s_idx - 1]
                alpha_bar_s = self.alpha_bar[s_idx - 1]
                mu = (x_s - beta_s / torch.sqrt(1.0 - alpha_bar_s + 1e-8) * eps_hat) / torch.sqrt(alpha_s)
                if s_idx > 1:
                    alpha_bar_prev = self.alpha_bar[s_idx - 2]
                    var = beta_s * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_s + 1e-8)
                    x_s = mu + torch.sqrt(var) * torch.randn_like(x_s)
                else:
                    x_s = mu
                x_s = x_s * (1.0 - obs_mask) + x0_full * obs_mask
            fut = x_s[:, self.history_len :]                      # (bsz*kc, Hp, N, C)
            groups.append(fut.reshape(bsz, kc, *fut.shape[1:]))   # (bsz, kc, Hp, N, C)
            remaining -= kc
        return torch.cat(groups, dim=1)                           # (bsz, n_samples, Hp, N, C)
