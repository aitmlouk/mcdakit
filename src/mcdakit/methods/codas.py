"""CODAS — Combinative Distance-based Assessment.

Keshavarz Ghorabaee et al. (2016).

The method takes the matrix as measured and resolves criterion direction
itself, so it is never handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._common import prepared
from .base import Method, ScoringContext, Wants


def codas(data, weights, directions: Sequence[str], tau: float = 0.02):
    """CODAS — COmbinative Distance-based ASsessment. Higher is better.

    Alternatives are ranked by Euclidean distance from the worst possible
    point, with Taxicab distance used to separate any pair whose Euclidean
    distances are within ``tau`` of each other. The threshold exists because
    Euclidean distance alone leaves near-ties that are arbitrary rather than
    meaningful.

    Keshavarz Ghorabaee, Zavadskas, Turskis and Antucheviciene, *A new
    combinative distance-based assessment (CODAS) method for multi-criteria
    decision-making*, Economic Computation and Economic Cybernetics Studies
    and Research 50(3), 2016.
    """
    data, w, is_cost = prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    low = data.min(axis=0)
    high = data.max(axis=0)
    safe_high = np.where(high == 0, 1.0, high)
    safe_data = np.where(data == 0, 1.0, data)
    normalised = np.where(is_cost, low / safe_data, data / safe_high)
    weighted = normalised * w

    negative_ideal = weighted.min(axis=0)
    euclidean = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    taxicab = np.abs(weighted - negative_ideal).sum(axis=1)

    n = data.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gap = euclidean[i] - euclidean[j]
            # Within tau the Euclidean distances are treated as tied, and the
            # Taxicab distance breaks the tie instead.
            psi = 1.0 if abs(gap) >= tau else 0.0
            scores[i] += gap + psi * (taxicab[i] - taxicab[j])
    return scores.astype(float)


class Codas(Method):
    """CODAS as a registered method."""

    name = "codas"
    summary = "Euclidean distance from the worst point, Taxicab breaking ties."
    citation = "Keshavarz Ghorabaee et al. (2016)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return codas(ctx.data, ctx.weights, ctx.directions, **ctx.opts)
