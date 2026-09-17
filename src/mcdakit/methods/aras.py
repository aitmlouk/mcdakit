"""ARAS — Additive Ratio Assessment.

Zavadskas and Turskis (2010).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import McdaError
from ._common import prepared
from .base import Method, ScoringContext, Wants


def aras(data, weights, directions: Sequence[str]) -> np.ndarray:
    """ARAS — Additive Ratio ASsessment. Higher is better.

    An optimal alternative is constructed from the best value available on
    each criterion, appended to the matrix, and every alternative is scored as
    a ratio of its weighted sum to the optimum's. The result is therefore a
    degree of utility in ``(0, 1]``, where 1 is attainable only by matching
    the best value on every criterion at once.

    Zavadskas and Turskis, *A new additive ratio assessment (ARAS) method in
    multicriteria decision-making*, Technological and Economic Development of
    Economy 16(2), 2010.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])
    if np.any(data <= 0):
        raise McdaError(
            "ARAS requires strictly positive values: cost criteria are "
            "inverted before normalisation, which a zero makes undefined."
        )

    # Cost criteria are inverted so that every column can be read as a
    # benefit, then the optimum is simply the column maximum.
    oriented = np.where(is_cost, 1.0 / data, data)
    optimal = oriented.max(axis=0)
    extended = np.vstack([optimal, oriented])

    normalised = extended / extended.sum(axis=0)
    utility = (normalised * w).sum(axis=1)
    return (utility[1:] / utility[0]).astype(float)


class Aras(Method):
    """ARAS as a registered method."""

    name = "aras"
    summary = "Additive ratio assessment against a constructed optimum."
    citation = "Zavadskas and Turskis (2010)"
    wants = Wants.RAW
    requires_positive = True

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return aras(ctx.data, ctx.weights, ctx.directions)
