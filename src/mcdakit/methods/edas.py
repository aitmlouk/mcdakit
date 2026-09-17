"""EDAS — Evaluation based on Distance from Average Solution.

Keshavarz Ghorabaee et al. (2015).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._common import prepared
from .base import Method, ScoringContext, Wants


def edas(data, weights, directions: Sequence[str]) -> np.ndarray:
    """EDAS — Evaluation based on Distance from Average Solution.

    Higher is better. Each alternative is measured by how far it rises above
    the column average and how far it falls below it, rather than against an
    ideal. That suits problems where the alternatives are broadly comparable
    and an ideal point would be a fiction.

    Keshavarz Ghorabaee, Zavadskas, Olfat and Turskis, *Multi-criteria
    inventory classification using a new method of evaluation based on
    distance from average solution*, Informatica 26(3), 2015.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    average = data.mean(axis=0)
    safe_average = np.where(average == 0, 1.0, average)

    positive = np.where(
        is_cost, (average - data) / safe_average, (data - average) / safe_average
    )
    negative = np.where(
        is_cost, (data - average) / safe_average, (average - data) / safe_average
    )
    positive = np.clip(positive, 0.0, None)
    negative = np.clip(negative, 0.0, None)

    weighted_positive = (positive * w).sum(axis=1)
    weighted_negative = (negative * w).sum(axis=1)

    best_positive = weighted_positive.max() or 1.0
    best_negative = weighted_negative.max() or 1.0
    normalised_positive = weighted_positive / best_positive
    normalised_negative = 1.0 - weighted_negative / best_negative
    return ((normalised_positive + normalised_negative) / 2.0).astype(float)


class Edas(Method):
    """EDAS as a registered method."""

    name = "edas"
    summary = "Distance above and below the average solution."
    citation = "Keshavarz Ghorabaee et al. (2015)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return edas(ctx.data, ctx.weights, ctx.directions)
