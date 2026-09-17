"""COPRAS — COmplex PRoportional ASsessment.

Zavadskas, Kaklauskas and Sarka, *The new method of multicriteria complex
proportional assessment of projects*, Technological and Economic Development
of Economy 1(3), 1994.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..normalization import normalize_weights
from ..types import McdaError
from .base import Method, ScoringContext, Wants


def copras(
    data: np.ndarray,
    weights: np.ndarray,
    directions: Sequence[str],
) -> np.ndarray:
    """Proportional assessment of benefit against cost. Higher is better.

    Sum-normalise each column, weight it, and total the benefit and cost
    contributions separately:

    .. math::

        S^{+}_i = \\sum_{j \\in benefit} w_j r_{ij}
        \\qquad
        S^{-}_i = \\sum_{j \\in cost} w_j r_{ij}
        \\qquad
        Q_i = S^{+}_i +
              \\frac{S^{-}_{min} \\sum_i S^{-}_i}
                   {S^{-}_i \\sum_i (S^{-}_{min} / S^{-}_i)}

    The second term is the point of the method: it is *inversely* proportional
    to an alternative's cost total, so a smaller cost raises the score. An
    implementation that adds :math:`S^{-}` directly would reward expense
    instead of penalising it, and would still produce a plausible-looking
    ranking.

    Unlike most methods here, COPRAS resolves criterion direction itself and
    so takes the matrix as measured. Values must be non-negative, which
    sum-normalisation requires; the method is defined for ratio-scale data.

    With no cost criterion the second term is undefined and the method reduces
    to a weighted sum, which is what is returned.
    """
    data = np.asarray(data, dtype=float)
    if np.any(data < 0):
        # Sum-normalisation is defined for ratio-scale data. A negative value
        # makes the column total meaningless and yields a NaN that would sort
        # unpredictably, so it is refused rather than propagated.
        raise McdaError(
            "COPRAS requires non-negative values: it normalises each "
            "criterion by its column total, which a negative value makes "
            "meaningless. Rescale the criterion, or use a method that "
            "tolerates negatives such as topsis or spotis."
        )
    w = normalize_weights(weights)
    if w is None:
        return np.zeros(data.shape[0])

    totals = data.sum(axis=0)
    normalised = data / np.where(totals == 0, 1.0, totals)
    weighted = normalised * w

    is_cost = np.asarray(directions) == "cost"
    benefit_total = weighted[:, ~is_cost].sum(axis=1)

    if not is_cost.any():
        # No cost criterion: the proportional term has nothing to work on.
        return benefit_total.astype(float)

    cost_total = weighted[:, is_cost].sum(axis=1)
    # An alternative costing nothing would divide by zero. Substituting the
    # smallest positive total keeps it ranked first on cost without producing
    # an infinity that would dominate every other criterion.
    safe = np.where(
        cost_total == 0, np.min(cost_total[cost_total > 0], initial=1.0), cost_total
    )

    minimum = safe.min()
    proportional = (minimum * safe.sum()) / (safe * np.sum(minimum / safe))
    return (benefit_total + proportional).astype(float)


class Copras(Method):
    """COPRAS as a registered method."""

    name = "copras"
    summary = "Complex proportional assessment of benefit against cost."
    citation = "Zavadskas, Kaklauskas and Sarka (1994)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return copras(ctx.data, ctx.weights, ctx.directions)
