"""COCOSO — Combined Compromise Solution.

Yazdani et al. (2019).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._common import prepared
from .base import Method, ScoringContext, Wants


def cocoso(data, weights, directions: Sequence[str], lambda_: float = 0.5):
    """COCOSO — COmbined COmpromise SOlution. Higher is better.

    Three aggregation strategies are computed --- an arithmetic mean of the
    weighted sum and weighted product, a sum of relative scores, and a
    balanced compromise --- and combined. Using three rather than one is the
    method's premise: an alternative that leads on all three is robust to the
    choice of aggregation, and one that leads on only a single strategy is
    visibly not.

    Yazdani, Zarate, Zavadskas and Turskis, *A combined compromise solution
    (CoCoSo) method for multi-criteria decision-making problems*, Management
    Decision 57(9), 2019.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    low = data.min(axis=0)
    high = data.max(axis=0)
    span = np.where(high - low == 0, 1.0, high - low)
    normalised = np.where(is_cost, (high - data) / span, (data - low) / span)

    weighted_sum = (normalised * w).sum(axis=1)
    # 0**0 is 1 in numpy, which is the reading wanted here: a criterion with
    # no weight contributes nothing to the product.
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted_product = np.nansum(
            np.where(normalised > 0, normalised**w, 0.0), axis=1
        )

    total = weighted_sum + weighted_product
    # An alternative worst on every criterion normalises to zero throughout,
    # which makes these minima and totals zero. Dividing then yields inf or
    # NaN, and such an alternative is neither exotic nor invalid; substituting
    # one leaves the ratio at its natural value rather than propagating a
    # non-finite score through the whole ranking.
    strategy_a = total / (total.sum() or 1.0)
    strategy_b = weighted_sum / (weighted_sum.min() or 1.0) + weighted_product / (
        weighted_product.min() or 1.0
    )
    scale = lambda_ * weighted_sum.max() + (1 - lambda_) * weighted_product.max()
    strategy_c = (lambda_ * weighted_sum + (1 - lambda_) * weighted_product) / (
        scale or 1.0
    )

    combined = (strategy_a * strategy_b * strategy_c) ** (1 / 3) + (
        strategy_a + strategy_b + strategy_c
    ) / 3
    return combined.astype(float)


class Cocoso(Method):
    """COCOSO as a registered method."""

    name = "cocoso"
    summary = "Combined compromise of three aggregation strategies."
    citation = "Yazdani, Zarate, Zavadskas and Turskis (2019)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return cocoso(ctx.data, ctx.weights, ctx.directions, **ctx.opts)
