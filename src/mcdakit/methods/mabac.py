"""MABAC — Multi-Attributive Border Approximation area Comparison.

Pamucar and Cirovic (2015).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._common import prepared
from .base import Method, ScoringContext, Wants


def mabac(data, weights, directions: Sequence[str]) -> np.ndarray:
    """MABAC — Multi-Attributive Border Approximation area Comparison.

    Higher is better, and the sign is meaningful: a positive total places an
    alternative in the upper approximation area, above the geometric mean of
    the field, and a negative total places it below. Unlike a closeness
    coefficient, the zero point is interpretable.

    Pamucar and Cirovic, *The selection of transport and handling resources in
    logistics centers using Multi-Attributive Border Approximation area
    Comparison (MABAC)*, Expert Systems with Applications 42(6), 2015.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    low = data.min(axis=0)
    high = data.max(axis=0)
    span = np.where(high - low == 0, 1.0, high - low)
    normalised = np.where(is_cost, (high - data) / span, (data - low) / span)

    weighted = w * (normalised + 1.0)
    border = np.prod(weighted, axis=0) ** (1.0 / data.shape[0])
    return (weighted - border).sum(axis=1).astype(float)


class Mabac(Method):
    """MABAC as a registered method."""

    name = "mabac"
    summary = "Distance from the border approximation area; sign is meaningful."
    citation = "Pamucar and Cirovic (2015)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return mabac(ctx.data, ctx.weights, ctx.directions)
