"""Methods that score against a reference point or an average.

Six methods sharing a shape: normalise the matrix, then compare each
alternative with something --- an ideal alternative, the column average, a
border approximation --- rather than only with the other alternatives. They
are grouped here because they differ mainly in the choice of reference, and
seeing that choice side by side is more useful than six near-identical files.

Every function takes the matrix as measured and resolves criterion direction
itself, so none of them is handed an oriented matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..normalization import normalize_weights
from ..types import McdaError
from .base import Method, ScoringContext, Wants


def _prepared(data, weights, directions):
    """Shared preamble: validate, normalise the weights, read the directions."""
    data = np.asarray(data, dtype=float)
    w = normalize_weights(weights)
    is_cost = np.asarray(directions) == "cost"
    return data, w, is_cost


# --------------------------------------------------------------------------
# ARAS
# --------------------------------------------------------------------------


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
    data, w, is_cost = _prepared(data, weights, directions)
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


# --------------------------------------------------------------------------
# COCOSO
# --------------------------------------------------------------------------


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
    data, w, is_cost = _prepared(data, weights, directions)
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


# --------------------------------------------------------------------------
# CODAS
# --------------------------------------------------------------------------


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
    data, w, is_cost = _prepared(data, weights, directions)
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


# --------------------------------------------------------------------------
# EDAS
# --------------------------------------------------------------------------


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
    data, w, is_cost = _prepared(data, weights, directions)
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


# --------------------------------------------------------------------------
# MABAC
# --------------------------------------------------------------------------


def mabac(data, weights, directions: Sequence[str]) -> np.ndarray:
    """MABAC — Multi-Attributive Border Approximation area Comparison.

    Higher is better, and the sign is meaningful: a positive total places an
    alternative in the upper approximation area, above the geometric mean of
    the field, and a negative total places it below. Unlike a closeness
    coefficient, the zero point is interpretable.

    Pamucar and Cirovic, *The selection of transport and handling resources in
    logistics centers using Multi-Attributive Border Approximation area
    Comparison (MABAC)*, Expert Systems with Applications 42(6), 2015.
    """
    data, w, is_cost = _prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    low = data.min(axis=0)
    high = data.max(axis=0)
    span = np.where(high - low == 0, 1.0, high - low)
    normalised = np.where(is_cost, (high - data) / span, (data - low) / span)

    weighted = w * (normalised + 1.0)
    border = np.prod(weighted, axis=0) ** (1.0 / data.shape[0])
    return (weighted - border).sum(axis=1).astype(float)


# --------------------------------------------------------------------------
# MARCOS
# --------------------------------------------------------------------------


def marcos(data, weights, directions: Sequence[str]) -> np.ndarray:
    """MARCOS — Measurement of Alternatives and Ranking according to
    COmpromise Solution. Higher is better.

    Both an ideal and an anti-ideal alternative are appended to the matrix,
    and each alternative is scored by its utility relative to the two at once.
    Bounding the problem from both ends is what distinguishes it from methods
    referencing only the ideal.

    Stevic, Pamucar, Puska and Chatterjee, *Sustainable supplier selection in
    healthcare industries using a new MCDM method: MARCOS*, Computers &
    Industrial Engineering 140, 2020.
    """
    data, w, is_cost = _prepared(data, weights, directions)
    if w is None:
        return np.zeros(data.shape[0])

    ideal = np.where(is_cost, data.min(axis=0), data.max(axis=0))
    anti_ideal = np.where(is_cost, data.max(axis=0), data.min(axis=0))
    extended = np.vstack([anti_ideal, data, ideal])

    safe_ideal = np.where(ideal == 0, 1.0, ideal)
    safe_extended = np.where(extended == 0, 1.0, extended)
    normalised = np.where(is_cost, safe_ideal / safe_extended, extended / safe_ideal)

    utility = (normalised * w).sum(axis=1)
    anti_utility, ideal_utility = utility[0], utility[-1]
    alternatives = utility[1:-1]

    k_minus = alternatives / (anti_utility or 1.0)
    k_plus = alternatives / (ideal_utility or 1.0)

    total = k_plus + k_minus
    f_plus = k_minus / total
    f_minus = k_plus / total
    denominator = 1.0 + (1.0 - f_plus) / f_plus + (1.0 - f_minus) / f_minus
    return (total / denominator).astype(float)


# --------------------------------------------------------------------------
# Registered methods
# --------------------------------------------------------------------------


class Aras(Method):
    """ARAS as a registered method."""

    name = "aras"
    summary = "Additive ratio assessment against a constructed optimum."
    citation = "Zavadskas and Turskis (2010)"
    wants = Wants.RAW
    requires_positive = True

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return aras(ctx.data, ctx.weights, ctx.directions)


class Cocoso(Method):
    """COCOSO as a registered method."""

    name = "cocoso"
    summary = "Combined compromise of three aggregation strategies."
    citation = "Yazdani, Zarate, Zavadskas and Turskis (2019)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return cocoso(ctx.data, ctx.weights, ctx.directions, **ctx.opts)


class Codas(Method):
    """CODAS as a registered method."""

    name = "codas"
    summary = "Euclidean distance from the worst point, Taxicab breaking ties."
    citation = "Keshavarz Ghorabaee et al. (2016)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return codas(ctx.data, ctx.weights, ctx.directions, **ctx.opts)


class Edas(Method):
    """EDAS as a registered method."""

    name = "edas"
    summary = "Distance above and below the average solution."
    citation = "Keshavarz Ghorabaee et al. (2015)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return edas(ctx.data, ctx.weights, ctx.directions)


class Mabac(Method):
    """MABAC as a registered method."""

    name = "mabac"
    summary = "Distance from the border approximation area; sign is meaningful."
    citation = "Pamucar and Cirovic (2015)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return mabac(ctx.data, ctx.weights, ctx.directions)


class Marcos(Method):
    """MARCOS as a registered method."""

    name = "marcos"
    summary = "Utility relative to both an ideal and an anti-ideal alternative."
    citation = "Stevic, Pamucar, Puska and Chatterjee (2020)"
    wants = Wants.RAW

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return marcos(ctx.data, ctx.weights, ctx.directions)
