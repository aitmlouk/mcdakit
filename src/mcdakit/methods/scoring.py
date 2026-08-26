"""Additive scoring methods: simple, weighted, and SAW.

All three take an already-oriented matrix — see :mod:`mcdakit.orientation` —
and return a score per option where higher is better.
"""

from __future__ import annotations

import numpy as np


def simple_scoring(data: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """Sum of raw scores per option, ignoring weights entirely.

    Useful only as a baseline and when every criterion is on the same scale.
    On mixed units it is meaningless — a price in thousands drowns a rating out
    of ten — and ``weighted_scoring`` should be used instead. ``weights`` is
    accepted and ignored so every method shares one signature.
    """
    return np.sum(data, axis=1).astype(float)


def weighted_scoring(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sum over min-max normalised scores.

    Each criterion is first scaled to ``[0, 1]`` across the options being
    compared, then weighted. Without that step the criterion with the largest
    numbers dominates regardless of its weight: a price spanning ~900 against a
    quality rating spanning ~3 means a 40% price weight behaves like 99%.

    ``r_ij = (x_ij - min_j) / (max_j - min_j)``, then ``sum_j w_j * r_ij``.

    A criterion every option scores identically carries no information; it is
    given a flat 1.0 rather than dividing by zero.
    """
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.zeros(data.shape[0])
    normalized_weights = weights / w_sum

    col_min = np.min(data, axis=0)
    col_max = np.max(data, axis=0)
    span = col_max - col_min
    span[span == 0] = 1.0
    normalized = (data - col_min) / span
    return (normalized @ normalized_weights).astype(float)


def saw(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Simple Additive Weighting (Churchman and Ackoff, 1954).

    ``r_ij = x_ij / max_j``, then ``sum_j w_j * r_ij``. Differs from
    ``weighted_scoring`` in normalising by the column maximum alone rather than
    the observed range, which keeps ratios between values intact but means the
    worst option is not pinned to zero.
    """
    w_sum = np.sum(weights)
    if w_sum == 0:
        return np.zeros(data.shape[0])
    w = weights / w_sum

    col_max = np.max(data, axis=0)
    col_max = np.where(col_max == 0, 1.0, col_max)
    normalized = data / col_max

    return (normalized @ w).astype(float)
