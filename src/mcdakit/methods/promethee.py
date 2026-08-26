"""PROMETHEE II — Preference Ranking Organization METHod for Enrichment
Evaluation.

Brans and Vincke, *A Preference Ranking Organisation Method: The PROMETHEE
Method for Multiple Criteria Decision-Making*, Management Science 31(6), 1985.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Method, ScoringContext

SHAPES = ("usual", "ushape", "vshape", "level", "linear", "gaussian")


def preference_degree(
    difference: float,
    shape: str = "usual",
    q: float = 0.0,
    p: float = 0.0,
    s: float = 0.0,
) -> float:
    """How strongly one option is preferred over another on one criterion.

    Brans and Vincke define six shapes. The default, ``usual``, treats any
    difference however small as total preference — so 4200 beats 4201 exactly
    as decisively as it beats 9000. That is right for a rating out of ten and
    wrong for money and days, which is why the others exist.

    ============  ========  ===============================================
    ``usual``     Type I    step at zero
    ``ushape``    Type II   indifferent up to ``q``, then total
    ``vshape``    Type III  linear from zero to ``p``
    ``level``     Type IV   indifferent to ``q``, half up to ``p``, then total
    ``linear``    Type V    indifferent to ``q``, linear to ``p``, then total
    ``gaussian``  Type VI   smooth, governed by ``s``
    ============  ========  ===============================================

    ``difference`` is (this option) minus (the other) *after* cost criteria
    have been oriented, so a difference at or below zero is never a preference.
    """
    if difference <= 0:
        return 0.0

    if shape == "ushape":
        return 1.0 if difference > q else 0.0

    if shape == "vshape":
        if p <= 0:
            return 1.0
        return 1.0 if difference > p else difference / p

    if shape == "level":
        if difference <= q:
            return 0.0
        if p <= q:
            return 1.0
        return 1.0 if difference > p else 0.5

    if shape == "linear":
        if difference <= q:
            return 0.0
        if p <= q:
            return 1.0
        if difference > p:
            return 1.0
        return (difference - q) / (p - q)

    if shape == "gaussian":
        if s <= 0:
            return 1.0
        return 1.0 - float(np.exp(-(difference**2) / (2.0 * s**2)))

    return 1.0


def preference_matrix(
    data: np.ndarray,
    weights: np.ndarray,
    shapes: Sequence | None = None,
) -> np.ndarray:
    """Aggregated preference of every option over every other.

    ``shapes`` is one ``(shape, q, p, s)`` tuple per criterion; omitting it
    uses the usual criterion throughout.

    Normalising by the weight total keeps the flows in ``[-1, 1]`` whatever
    scale the weights were entered on, so 40/30/20/10 and 0.4/0.3/0.2/0.1
    produce identical flows rather than differing by a factor of a hundred.
    """
    n_alternatives, n_criteria = data.shape
    matrix = np.zeros((n_alternatives, n_alternatives))

    weight_total = float(np.sum(weights)) or 1.0

    for i in range(n_alternatives):
        for j in range(n_alternatives):
            if i == j:
                continue
            pref_sum = 0.0
            for k in range(n_criteria):
                difference = data[i, k] - data[j, k]
                if shapes:
                    shape, q, p, s = shapes[k]
                    degree = preference_degree(difference, shape, q, p, s)
                else:
                    degree = 1.0 if difference > 0 else 0.0
                pref_sum += weights[k] * degree
            matrix[i, j] = pref_sum / weight_total

    return matrix


def flows(matrix: np.ndarray):
    """``(positive, negative, net)`` flows from a preference matrix.

    The positive flow is how much an option outranks the field, the negative
    flow how much the field outranks it, and the net flow their difference —
    PROMETHEE II's complete ranking.
    """
    positive = np.sum(matrix, axis=1)
    negative = np.sum(matrix, axis=0)
    return positive, negative, positive - negative


def promethee(
    data: np.ndarray,
    weights: np.ndarray,
    shapes: Sequence | None = None,
) -> np.ndarray:
    """Net flows, in ``[-1, 1]``, higher is better."""
    return flows(preference_matrix(data, weights, shapes))[2].astype(float)


def shapes_from(criteria) -> tuple | None:
    """``(shape, q, p, s)`` per criterion, or ``None`` if all are the default.

    Returning ``None`` when nothing was configured keeps PROMETHEE on its
    simple path, so a problem that never touched preference functions scores
    bit-identically to one built before they existed.
    """
    shapes = [(c.preference_shape, c.q, c.p, c.s) for c in criteria]
    if all(shape == "usual" for shape, _q, _p, _s in shapes):
        return None
    return tuple(shapes)


class Promethee(Method):
    """PROMETHEE II as a registered method.

    Reads each criterion's ``preference_shape``, ``q``, ``p`` and ``s`` off the
    problem, so the six preference functions need no extra plumbing at the
    call site.
    """

    name = "promethee"
    summary = "PROMETHEE II net flows, with six preference-function shapes."
    citation = "Brans and Vincke (1985)"

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return promethee(ctx.data, ctx.weights, shapes_from(ctx.criteria), **ctx.opts)
