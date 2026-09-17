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
from dataclasses import dataclass

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
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
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
            raise McdaError(f"Judgement ({i}, {j}) is out of range for {n} criteria.")
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


def consistency_ratio(matrix: np.ndarray, weights: np.ndarray | None = None) -> float:
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


@dataclass(frozen=True)
class AhpResult:
    """A full AHP hierarchy, and how coherent the judgements behind it were.

    Attributes
    ----------
    scores:
        One priority per alternative, summing to one. Higher is better.
    weights:
        The criterion weights derived from the criteria comparison matrix.
    local_priorities:
        Alternatives by criteria: each column is the alternatives' priorities
        under one criterion.
    consistency:
        ``{"criteria": ratio, criterion_name: ratio, ...}``. Reported for
        every matrix separately, because a hierarchy is only as sound as its
        least coherent judgement set and an average would hide exactly that.
    inconsistent:
        Names of the matrices exceeding :data:`CONSISTENCY_LIMIT`. Empty when
        every judgement set holds together.
    """

    scores: np.ndarray
    weights: np.ndarray
    local_priorities: np.ndarray
    names: tuple
    labels: tuple
    consistency: dict
    inconsistent: tuple

    @property
    def ranking(self) -> list:
        """``[(label, score), ...]`` best first."""
        order = sorted(range(len(self.scores)), key=lambda i: -self.scores[i])
        return [(self.labels[i], float(self.scores[i])) for i in order]

    @property
    def winner(self) -> str:
        return self.ranking[0][0]

    def __str__(self) -> str:
        lines = ["AHP ranking:"]
        label_width = max((len(label) for label, _ in self.ranking), default=0) + 2
        for position, (label, score) in enumerate(self.ranking, start=1):
            lines.append(f"  {position}. {label:<{label_width}}{score:.4f}")
        lines.append("  consistency ratios:")
        name_width = max((len(n) for n in self.consistency), default=0) + 2
        for name, ratio in self.consistency.items():
            flag = "  EXCEEDS 0.10" if ratio > CONSISTENCY_LIMIT else ""
            lines.append(f"    {name:<{name_width}}{ratio:.4f}{flag}")
        if self.inconsistent:
            verb = "contradicts" if len(self.inconsistent) == 1 else "contradict"
            lines.append(
                f"  ! {', '.join(self.inconsistent)} {verb} itself; revisit "
                f"those judgements before relying on the ranking"
                if len(self.inconsistent) == 1
                else f"  ! {', '.join(self.inconsistent)} {verb} themselves; "
                f"revisit those judgements before relying on the ranking"
            )
        return "\n".join(lines)


def ahp_rank(
    criteria_comparisons,
    alternative_comparisons: Sequence,
    names: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
) -> AhpResult:
    """Rank alternatives by pairwise comparison throughout the hierarchy.

    This is AHP as Saaty defined it, and it differs from every other method in
    this package in what it asks for. There is no decision matrix: instead the
    alternatives are compared against each other, pair by pair, separately
    under each criterion. That is what makes the method usable on criteria
    nobody can measure --- a supplier's reputation, the quality of a design ---
    where asking *is A better than B on this, and by how much* is answerable
    and asking for a number is not.

    The price is quadratic: :math:`n(n-1)/2` judgements per criterion, plus
    the criteria themselves. The compensation is that the judgements can be
    checked. A person who says A is three times B, B three times C, and then
    A twice C has contradicted themselves, and the consistency ratio detects
    it without anyone knowing what the right answer was.

    Parameters
    ----------
    criteria_comparisons:
        A reciprocal matrix comparing the criteria, or a ``{(i, j): ratio}``
        mapping of its upper triangle.
    alternative_comparisons:
        One comparison matrix per criterion, in the same order as the
        criteria, each comparing the alternatives under that criterion.
    names, labels:
        Criterion and alternative names, for the report.

    Returns
    -------
    AhpResult
        With the ranking, the weights, and a consistency ratio for every
        matrix. Inconsistency is reported and never enforced: whether a
        contradiction is fatal is the analyst's judgement, not the library's.

    Examples
    --------
    >>> import numpy as np
    >>> from mcdakit import ahp_rank
    >>> criteria = np.array([[1.0, 3.0], [1 / 3, 1.0]])
    >>> per_criterion = [
    ...     np.array([[1.0, 5.0], [1 / 5, 1.0]]),
    ...     np.array([[1.0, 1 / 3], [3.0, 1.0]]),
    ... ]
    >>> result = ahp_rank(criteria, per_criterion, labels=["A", "B"])
    >>> result.winner
    'A'
    """
    alternative_comparisons = list(alternative_comparisons)
    if not alternative_comparisons:
        raise McdaError(
            "ahp_rank needs one comparison matrix per criterion; none were given."
        )

    n_criteria = len(alternative_comparisons)
    if names is None:
        names = [f"Criterion {j + 1}" for j in range(n_criteria)]
    names = tuple(str(n) for n in names)
    if len(names) != n_criteria:
        raise McdaError(
            f"Got {len(names)} criterion names for {n_criteria} comparison matrices."
        )

    if isinstance(criteria_comparisons, dict):
        criteria_matrix = comparison_matrix(n_criteria, criteria_comparisons)
    else:
        criteria_matrix = np.asarray(criteria_comparisons, dtype=float)
    if criteria_matrix.shape != (n_criteria, n_criteria):
        raise McdaError(
            f"The criteria comparison matrix is {criteria_matrix.shape}, but "
            f"{n_criteria} comparison matrices were given, one per criterion."
        )

    weights = priorities(criteria_matrix)
    consistency = {"criteria": float(consistency_ratio(criteria_matrix, weights))}

    columns, n_alternatives = [], None
    for name, comparisons in zip(names, alternative_comparisons):
        matrix = np.asarray(comparisons, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise McdaError(
                f"The comparison matrix for {name!r} must be square, got "
                f"shape {matrix.shape}."
            )
        if n_alternatives is None:
            n_alternatives = matrix.shape[0]
        elif matrix.shape[0] != n_alternatives:
            raise McdaError(
                f"The comparison matrix for {name!r} compares "
                f"{matrix.shape[0]} alternatives, but earlier matrices "
                f"compared {n_alternatives}. Every criterion must rank the "
                f"same alternatives."
            )
        local = priorities(matrix)
        columns.append(local)
        consistency[name] = float(consistency_ratio(matrix, local))

    if labels is None:
        labels = [f"Option {i + 1}" for i in range(n_alternatives or 0)]
    labels = tuple(str(label) for label in labels)
    if len(labels) != n_alternatives:
        raise McdaError(
            f"Got {len(labels)} alternative labels for {n_alternatives} alternatives."
        )

    local_priorities = np.column_stack(columns)
    scores = local_priorities @ weights

    inconsistent = tuple(
        name for name, ratio in consistency.items() if ratio > CONSISTENCY_LIMIT
    )
    return AhpResult(
        scores=scores.astype(float),
        weights=weights,
        local_priorities=local_priorities,
        names=names,
        labels=labels,
        consistency=consistency,
        inconsistent=inconsistent,
    )
