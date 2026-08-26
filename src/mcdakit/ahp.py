"""AHP — deriving weights from pairwise comparisons.

Saaty, *The Analytic Hierarchy Process*, McGraw-Hill, 1980.

Weights are the softest input in any MCDA model, and typing them directly is
the worst way to produce them: people cannot reliably say a criterion is worth
0.35 rather than 0.40, but they can say price matters somewhat more than
quality. AHP turns those judgements into weights and — the part that makes it
worth the extra questions — measures whether the judgements contradict each
other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import McdaError

#: Saaty's fundamental scale, as ratio and meaning.
SAATY_SCALE = {
    1 / 9: "Extremely less important",
    1 / 7: "Very strongly less important",
    1 / 5: "Strongly less important",
    1 / 3: "Moderately less important",
    1.0: "Equally important",
    3.0: "Moderately more important",
    5.0: "Strongly more important",
    7.0: "Very strongly more important",
    9.0: "Extremely more important",
}

#: Saaty's random consistency index by matrix order: the average consistency
#: index of a randomly filled n x n reciprocal matrix. The consistency ratio is
#: CI / RI, which is what makes it comparable across problem sizes.
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
    7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
    13: 1.56, 14: 1.57, 15: 1.59,
}

#: Saaty's rule of thumb: above this, at least one judgement contradicts the
#: others badly enough that the derived weights should not be relied on.
CONSISTENCY_LIMIT = 0.10


def comparison_matrix(n: int, judgements: dict) -> np.ndarray:
    """Build a reciprocal matrix from ``{(i, j): ratio}`` judgements.

    Only the upper triangle need be given: setting ``(i, j)`` to 3 sets
    ``(j, i)`` to 1/3 automatically, which is what makes the matrix reciprocal
    and the exercise tractable — n(n-1)/2 questions rather than n².
    Unjudged pairs default to 1 (equally important).
    """
    matrix = np.ones((n, n), dtype=float)
    for (i, j), value in judgements.items():
        if not (0 <= i < n and 0 <= j < n):
            raise McdaError(
                f"Judgement ({i}, {j}) is out of range for {n} criteria."
            )
        if i == j:
            raise McdaError("A criterion cannot be compared with itself.")
        value = float(value)
        if value <= 0:
            raise McdaError(
                f"Judgement ({i}, {j}) must be a positive ratio, got {value}."
            )
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value
    return matrix


def priorities(matrix: np.ndarray) -> np.ndarray:
    """Principal eigenvector by row geometric mean, normalised to sum to one.

    The geometric mean is Saaty's standard approximation and — unlike an
    iterative eigensolver — always returns something sensible for the small,
    well-formed matrices this is used on. It also keeps the numpy-only rule:
    no scipy needed.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise McdaError(
            f"A comparison matrix must be square, got shape {matrix.shape}."
        )
    geometric = np.prod(matrix, axis=1) ** (1.0 / matrix.shape[0])
    total = geometric.sum()
    if total == 0:
        return np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    return geometric / total


def consistency_ratio(matrix: np.ndarray, weights: np.ndarray = None) -> float:
    """Saaty's consistency ratio, ``CI / RI``.

    Zero for a perfectly consistent set of judgements — one where saying price
    is 3x quality and quality is 3x delivery is matched by saying price is 9x
    delivery. Returns 0.0 for orders below three, where a single judgement has
    nothing to contradict.
    """
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    if size < 3:
        return 0.0
    if weights is None:
        weights = priorities(matrix)

    weighted_sum = matrix @ weights
    lambda_max = float(np.mean(weighted_sum / weights))
    consistency_index = (lambda_max - size) / (size - 1)
    random_index = RANDOM_INDEX.get(size, 1.60)
    return consistency_index / random_index if random_index else 0.0


def ahp_weights(judgements, names: Sequence[str] | None = None) -> dict:
    """Weights from pairwise comparisons, with their consistency ratio.

    ``judgements`` is either a full reciprocal matrix, or a ``{(i, j): ratio}``
    mapping of the upper triangle — in which case ``names`` (or an explicit
    ``n``) is needed to size it.

    Returns ``{"weights", "consistency_ratio", "consistent", "names"}``.

    The consistency ratio is **reported, never enforced**. An inconsistent
    matrix is a signal to revisit the judgements, not an error: sometimes the
    inconsistency is real and the analyst wants to proceed anyway, and taking
    that decision away from them would be presumptuous.

    Examples
    --------
    >>> from mcdakit import ahp_weights
    >>> out = ahp_weights({(0, 1): 3, (0, 2): 9, (1, 2): 3},
    ...                   names=["Price", "Quality", "Delivery"])
    >>> round(out["consistency_ratio"], 6)
    0.0
    >>> [round(float(w), 4) for w in out["weights"]]
    [0.6923, 0.2308, 0.0769]
    """
    if isinstance(judgements, dict):
        if names is None:
            raise McdaError(
                "names are required when judgements are given as pairs, so "
                "the matrix size is known."
            )
        matrix = comparison_matrix(len(names), judgements)
    else:
        matrix = np.asarray(judgements, dtype=float)
        if names is None:
            names = [f"Criterion {i + 1}" for i in range(matrix.shape[0])]

    if matrix.shape[0] < 2:
        raise McdaError(
            "Pairwise comparison needs at least two criteria; with one there "
            "is nothing to compare."
        )
    if len(names) != matrix.shape[0]:
        raise McdaError(
            f"Got {len(names)} names for a {matrix.shape[0]}x"
            f"{matrix.shape[1]} comparison matrix."
        )

    weights = priorities(matrix)
    ratio = consistency_ratio(matrix, weights)

    return {
        "weights": weights,
        "consistency_ratio": ratio,
        "consistent": ratio <= CONSISTENCY_LIMIT,
        "names": list(names),
        "matrix": matrix,
    }
