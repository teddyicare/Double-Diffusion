"""
physics_ode.py
--------------
Parameterless graph diffusion ODE solver.

Implements the physics prior:
    dX/dt = -k * L * X

where L is the normalized graph Laplacian and k is a fixed (non-learned)
diffusion coefficient.

The analytical solution in the spectral domain is:
    X(t) = U * exp(-k * Λ * t) * Uᵀ * X(0)

where L = U Λ Uᵀ is the eigendecomposition.
This is more numerically stable and faster than numerical ODE integration
for the smooth diffusion equation, and is equivalent to the dopri5 solver
used in the original implementation.

For AirPhyNet's full advection-diffusion model, see airphynet.py.
"""

import torch
import torch.nn as nn
from typing import Optional


class PhysicsODESolver(nn.Module):
    """
    Closed-form solver for dX/dt = -k * L * X using spectral decomposition.

    X(t) = U * diag(exp(-k * λ * t)) * Uᵀ * X(0)

    This is:
    1. Parameterless: k is a fixed scalar, not learned.
    2. Interpretable: the filter exp(-k * λ * t) is a known graph spectral
       low-pass filter — high graph frequencies (large λ) decay exponentially
       faster than low frequencies.
    3. Fast: O(N²) per forward pass (two matrix multiplications).

    Parameters
    ----------
    eigenvalues  : (N,) tensor of Laplacian eigenvalues (precomputed)
    eigenvectors : (N, N) tensor of Laplacian eigenvectors (precomputed)
    k            : diffusion coefficient (scalar, not learned)
                   k=0.1 for PM2.5 following AirPhyNet;
                   tuned via grid search for other pollutants/domains
    """

    def __init__(
        self,
        eigenvalues:  torch.Tensor,
        eigenvectors: torch.Tensor,
        k:            float = 0.1,
    ):
        super().__init__()
        # Register as buffers — moved with .to(device) but not trained
        self.register_buffer('eigenvalues',  eigenvalues.float())   # (N,)
        self.register_buffer('eigenvectors', eigenvectors.float())  # (N, N)
        self.k = k

    @torch.no_grad()
    def forward(
        self,
        x_last:     torch.Tensor,    # (B, N, C)  last observed values
        n_steps:    int,             # number of future steps to predict
        dt:         float = 1.0,    # time step size (1 hour = 1.0)
    ) -> torch.Tensor:
        """
        Generate a physics-based forecast for the next n_steps time steps.

        For each future step h = 1, ..., n_steps:
            X(h * dt) = U * diag(exp(-k * λ * h * dt)) * Uᵀ * X(0)

        Parameters
        ----------
        x_last  : (B, N, C) — last observed values (raw/unnormalized)
        n_steps : number of future steps
        dt      : time step duration (default 1.0 for hourly)

        Returns
        -------
        x_pred : (B, H', N, C) — predicted future states
        """
        B, N, C = x_last.shape
        device = x_last.device
        U   = self.eigenvectors   # (N, N)
        Lam = self.eigenvalues    # (N,)

        # Precompute spectral filter for each future step
        # filter[h, i] = exp(-k * λ_i * (h+1) * dt)
        steps = torch.arange(1, n_steps + 1, dtype=torch.float32, device=device)  # (H',)
        # (H', N): exp(-k * λ * h * dt)
        filters = torch.exp(
            -self.k * Lam.unsqueeze(0) * steps.unsqueeze(1) * dt
        )  # (H', N)

        # Transform X(0) to spectral domain: X̃(0) = Uᵀ X(0)
        # x_last: (B, N, C) — need (N, B*C) for matmul
        x0 = x_last.permute(1, 0, 2).reshape(N, B * C)   # (N, B*C)
        x0_spec = U.T @ x0                                 # (N, B*C)  graph Fourier transform

        # Apply filter at each future step
        # x0_spec: (N, B*C), filters[h]: (N,)
        # → x_spec_h = filters[h, :, None] * x0_spec → (N, B*C)
        # → x_h = U @ x_spec_h → (N, B*C)

        predictions = []
        for h in range(n_steps):
            filter_h = filters[h].unsqueeze(1)              # (N, 1)
            x_spec_h = filter_h * x0_spec                   # (N, B*C)
            x_h      = U @ x_spec_h                         # (N, B*C)  inverse GFT
            x_h      = x_h.reshape(N, B, C).permute(1, 0, 2)  # (B, N, C)
            predictions.append(x_h)

        x_pred = torch.stack(predictions, dim=1)  # (B, H', N, C)
        return x_pred

    @torch.no_grad()
    def spectral_filter_response(self, t: float = 1.0) -> torch.Tensor:
        """
        Return the spectral filter response H(λ) = exp(-k * λ * t) for all λ.
        Used for visualization and for the spectral energy analysis in the paper.

        Returns
        -------
        response : (N,) tensor, values in (0, 1], monotonically decreasing
        """
        return torch.exp(-self.k * self.eigenvalues * t)

    def extra_repr(self) -> str:
        N = self.eigenvalues.shape[0]
        return f"N={N}, k={self.k}"


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_residual(
    x_pred:   torch.Tensor,   # (B, H', N, C) physics prediction (unnormalized)
    x_future: torch.Tensor,   # (B, H', N, C) ground truth (unnormalized)
) -> torch.Tensor:
    """
    Compute the physics residual: Res = x_pred - x_future.

    In Resfusion, the forward process drifts toward Res instead of pure noise.
    The denoiser learns to predict the residual noise.

    Returns
    -------
    residual : (B, H', N, C)  same shape as inputs
    """
    return x_pred - x_future


def normalize_with_stats(
    x:    torch.Tensor,   # (*, C)
    mean: torch.Tensor,   # (C,)
    std:  torch.Tensor,   # (C,)
) -> torch.Tensor:
    """Normalize tensor using precomputed mean/std."""
    return (x - mean.to(x.device)) / std.to(x.device)


def denormalize_with_stats(
    x:    torch.Tensor,
    mean: torch.Tensor,
    std:  torch.Tensor,
) -> torch.Tensor:
    """Inverse normalization."""
    return x * std.to(x.device) + mean.to(x.device)
