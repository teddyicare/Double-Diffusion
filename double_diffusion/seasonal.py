"""Leakage-free seasonal index helper.

Several scripts build a "same time in a previous period" seasonal prior
(e.g. same-hour-yesterday for hourly data, same-5min-yesterday for traffic).
The naive implementation took a contiguous slice ``data[t - period : t - period
+ future_len]``. When ``future_len > period`` that slice runs *past* the
forecast origin ``t`` and copies ground-truth future values into the prior
(label leakage). This was present in models.py, diagnose_prior_k.py, and
seasonal_pool_baseline.py.

This module is the single source of truth for the correct indices. It is pure
Python (no torch) so every caller can share it without import-path or device
concerns.
"""

from __future__ import annotations


def previous_period_indices(start: int, period: int, future_len: int) -> list[int] | None:
    """Absolute time indices for a leakage-free same-phase seasonal lookup.

    The forecast target window is ``[start, start + future_len)``. For each
    horizon step ``j`` (0-indexed) we want the value at the *same phase*
    ``(start + j) mod period`` taken from the most recent prior period whose
    index is **strictly before** ``start`` (so no future ground truth leaks).

    For ``future_len <= period`` this is identical to the old contiguous slice
    ``data[start - period : start - period + future_len]`` (which already ended
    at or before ``start``). For ``future_len > period`` it tiles the last full
    period block ``[start - period, start)`` instead of reading past ``start``.

    Returns ``None`` when no complete prior period is available
    (``start <= period``), so callers can fall back to persistence.
    """
    if start - period < 0:
        return None
    # j // period = number of *extra* whole periods to step back so the index
    # stays < start. j in [0, period) -> back 1 period; [period, 2*period) -> 2; ...
    return [start + j - period * ((j // period) + 1) for j in range(future_len)]
