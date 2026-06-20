"""Split-conformal calibration for the predictive intervals.

The raw diffusion-ensemble 90% intervals are badly under-covered in the pilots
(~0.2-0.46 vs a 0.90 target). This module fits an additive interval widening on
the validation set (conformalized-quantile-regression style) so the test
intervals attain approximately nominal marginal coverage, with the standard
finite-sample split-conformal guarantee.

Widening is fit per (horizon-step, channel) because miscalibration grows with
lead time and varies by variable scale. The fit uses only the validation split;
nothing from the test split enters the correction.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def interval_bounds(samples_raw: torch.Tensor, lo: float = 0.05, hi: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower/upper ensemble-quantile bounds. samples_raw: (B, S, H', N, C)."""
    q_lo = torch.quantile(samples_raw, lo, dim=1)
    q_hi = torch.quantile(samples_raw, hi, dim=1)
    return q_lo, q_hi


@torch.no_grad()
def fit_conformal_widening(
    model,
    loader,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    n_samples: int,
    alpha: float = 0.10,
    reveal_prefix: int = 0,
    start_step=None,
    max_batches: int = 0,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Fit additive interval widening Q of shape (H', C) in RAW units.

    Nonconformity score per element: E = max(q_lo - y, y - q_hi) (positive when
    the target falls outside the raw interval). Q[h, c] is the conformal quantile
    of E at level (1-alpha)(1 + 1/n) over all validation (batch, node) pairs for
    that (step, channel). Test intervals are then [q_lo - Q, q_hi + Q].

    The widening MUST be fit with the same prediction configuration (reveal
    prefix and reverse start step) as the policy it will be applied to, because
    those change the interval scale. Returns None if the loader is empty.
    """
    mean = mean.to(device)
    std = std.to(device)
    collected: list[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        target = batch["x_future"].to(device)  # (B, H', N, C) normalized
        samples = model.predict(
            batch, mean, std, n_samples=n_samples, reveal_prefix=reveal_prefix, start_step=start_step
        )  # (B, S, H', N, C)
        samples_raw = samples * std + mean
        target_raw = target * std + mean
        q_lo, q_hi = interval_bounds(samples_raw)  # (B, H', N, C)
        score = torch.maximum(q_lo - target_raw, target_raw - q_hi)  # (B, H', N, C)
        # collapse (B, N) into one axis -> (B*N, H', C)
        score = score.permute(0, 2, 1, 3).reshape(-1, score.shape[1], score.shape[3])
        collected.append(score.cpu())
    if not collected:
        return None
    all_scores = torch.cat(collected, dim=0)  # (M, H', C)
    n = all_scores.shape[0]
    # Finite-sample split-conformal level; clamp to a valid quantile in [0, 1].
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(n, 1)))
    q = torch.quantile(all_scores, level, dim=0)  # (H', C); may be negative if over-covered
    return q
