"""MARCOS — Measurement of Alternatives and Ranking according to COmpromise Solution.

Stevic et al. (2020).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._common import prepared
from .base import Method, ScoringContext, Wants


def marcos(data, weights, directions: Sequence[str]) -> np.ndarray:
    """MARCOS — Measurement of Alternatives and Ranking according to
    COmpromise Solution. Higher is better.

    Both an ideal and an anti-ideal alternative are appended to the matrix,
    and each alternative is scored by its utility relative to the two at once.
    Bounding the problem from both ends is what distinguishes it from methods
    referencing only the ideal.

    Stevic, Pamucar, Puska and Chatterjee, *Sustainable supplier selection in
    healthcare industries using a new MCDM method: MARCOS*, Computers &
    Industrial Engineering 140, 2020.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    ideal = np.where(is_cost, data.min(axis=0), data.max(axis=0))
    anti_ideal = np.where(is_cost, data.max(axis=0), data.min(axis=0))
    extended = np.vstack([anti_ideal, data, ideal])

    safe_ideal = np.where(ideal == 0, 1.0, ideal)
    safe_extended = np.where(extended == 0, 1.0, extended)
    normalised = np.where(is_cost, safe_ideal / safe_extended, extended / safe_ideal)

    utility = (normalised * w).sum(axis=1)
    anti_utility, ideal_utility = utility[0], utility[-1]
    alternatives = utility[1:-1]

    k_minus = alternatives / (anti_utility or 1.0)
    k_plus = alternatives / (ideal_utility or 1.0)

    total = k_plus + k_minus
    f_plus = k_minus / total
    f_minus = k_plus / total
    denominator = 1.0 + (1.0 - f_plus) / f_plus + (1.0 - f_minus) / f_minus
    return (total / denominator).astype(float)


# --------------------------------------------------------------------------
# Registered methods


class Marcos(Method):
    """MARCOS as a registered method."""

    name = "marcos"
    summary = "Utility relative to both an ideal and an anti-ideal alternative."
    citation = "Stevic, Pamucar, Puska and Chatterjee (2020)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return marcos(ctx.data, ctx.weights, ctx.directions)
