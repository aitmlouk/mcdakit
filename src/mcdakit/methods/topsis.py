"""TOPSIS — Technique for Order of Preference by Similarity to Ideal Solution.

Hwang and Yoon, *Multiple Attribute Decision Making: Methods and
Applications*, Springer, 1981.
"""

from __future__ import annotations

import numpy as np

from ..normalization import normalize_weights
from .base import Method, ScoringContext


def topsis(
    data: np.ndarray, weights: np.ndarray, normalization: str = "vector"
) -> np.ndarray:
    """Closeness to the ideal solution, in ``[0, 1]``, higher is better.

    Vector-normalise (``r_ij = x_ij / sqrt(sum_i x_ij^2)``), weight, then take
    each option's Euclidean distance to the best and worst attainable points.
    The score is ``d- / (d+ + d-)``: the share of an option's total distance
    that lies on the good side.

    Normalisation is vector normalisation, Hwang and Yoon's original choice.
    This is worth stating because it is not universal: `pymcdm` defaults to
    min-max normalisation here, and the two give genuinely different scores on
    the same data. Neither is wrong — normalisation is a modelling decision,
    not a preprocessing detail — but comparing numbers across libraries
    without knowing which was used will mislead.

    Both ideal points are read off the options present, which is why TOPSIS
    reverses rank at a measurable rate — around 32% in this package's own
    benchmark when the last-placed option is removed. That is inherent to the
    method, not a defect in this implementation; use ``spotis`` when the
    ranking must be stable under a changing shortlist.
    """
    n_alt = data.shape[0]

    w = normalize_weights(weights)
    if w is None:
        return np.zeros(n_alt)

    from ..normalization import normalize as _normalize

    v = _normalize(data, normalization) * w

    ideal_best = np.max(v, axis=0)
    ideal_worst = np.min(v, axis=0)

    d_pos = np.sqrt(np.sum((v - ideal_best) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((v - ideal_worst) ** 2, axis=1))

    denominator = d_pos + d_neg
    denominator = np.where(denominator == 0, 1.0, denominator)
    return (d_neg / denominator).astype(float)


class Topsis(Method):
    """TOPSIS as a registered method."""

    name = "topsis"
    summary = "Closeness to the ideal solution, by vector normalisation."
    citation = "Hwang and Yoon (1981)"

    normalization = "vector"

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return topsis(ctx.data, ctx.weights, ctx.normalization)
