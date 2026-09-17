"""Preparation shared by the methods that resolve criterion direction themselves.

These methods take the matrix as measured rather than pre-oriented, so each
begins by validating it, normalising the weights and reading the directions.
Keeping that preamble in one place means a change to it cannot apply to five
methods and miss the sixth.
"""

from __future__ import annotations

import numpy as np

from ..normalization import normalize_weights


def prepared(data, weights, directions):
    """Validate, normalise the weights, and read the criterion directions.

    Returns ``(data, weights, is_cost)``. ``weights`` is ``None`` when they
    sum to zero, which each caller turns into a zero score vector.
    """
    data = np.asarray(data, dtype=float)
    w = normalize_weights(weights)
    is_cost = np.asarray(directions) == "cost"
    return data, w, is_cost
