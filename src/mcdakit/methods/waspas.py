"""WASPAS — Weighted Aggregated Sum Product Assessment.

Zavadskas, Turskis, Antucheviciene and Zakarevicius, *Optimization of weighted
aggregated sum product assessment*, Elektronika ir Elektrotechnika 122(6),
2012.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..normalization import normalize_weights
from ..types import McdaError
from .base import Method, ScoringContext, Wants


def waspas(
    data: np.ndarray,
    weights: np.ndarray,
    directions: Sequence[str],
    lambda_: float = 0.5,
) -> np.ndarray:
    """A blend of the weighted sum and weighted product models.

    .. math::

        Q_i = \\lambda \\sum_j w_j r_{ij}
              + (1 - \\lambda) \\prod_j r_{ij}^{\\,w_j}

    where :math:`r` is linearly normalised --- divided by the column maximum
    for a benefit criterion, and the column minimum divided by the value for a
    cost criterion.

    The two halves fail differently, which is the method's point. The sum is
    fully compensatory: a very poor score on one criterion can be offset by
    good scores elsewhere. The product is not, because a near-zero value drags
    the whole product down however good the rest are. Blending them at
    :math:`\\lambda = 0.5` splits the difference; :math:`\\lambda = 1` recovers
    the weighted sum and :math:`\\lambda = 0` the weighted product.

    Like COPRAS, the method resolves criterion direction itself and takes the
    matrix as measured. Values must be positive: the cost normalisation
    divides by them, and the product raises them to fractional powers.
    """
    data = np.asarray(data, dtype=float)
    if not 0.0 <= lambda_ <= 1.0:
        raise McdaError(
            f"lambda_ must lie between 0 and 1, where 1 is the weighted sum "
            f"and 0 the weighted product; got {lambda_}."
        )
    if np.any(data <= 0):
        raise McdaError(
            "WASPAS requires strictly positive values: it divides by them for "
            "cost criteria and raises them to fractional powers for the "
            "product term. Rescale the criterion, or use a method that "
            "tolerates zero such as topsis or spotis."
        )

    w = normalize_weights(weights)
    if w is None:
        return np.zeros(data.shape[0])

    is_cost = np.asarray(directions) == "cost"
    normalised = np.where(is_cost, data.min(axis=0) / data, data / data.max(axis=0))

    weighted_sum = (normalised * w).sum(axis=1)
    weighted_product = np.prod(normalised**w, axis=1)
    return (lambda_ * weighted_sum + (1.0 - lambda_) * weighted_product).astype(float)


def wpm(
    data: np.ndarray,
    weights: np.ndarray,
    directions: Sequence[str],
) -> np.ndarray:
    """The weighted product model, also called MEW.

    .. math::

        S_i = \\prod_j r_{ij}^{\\,w_j}

    Bridgman, *Dimensional Analysis* (1922); Miller and Starr, *Executive
    Decisions and Operations Research* (1969).

    Where the weighted sum lets a strength pay for a weakness, the product
    does not: one value near zero drags the whole score down however good the
    rest are. That non-compensatory behaviour is the reason to choose it, and
    it matters most when a criterion represents something disqualifying rather
    than merely undesirable --- a supplier who cannot meet a safety threshold
    is not redeemed by a low price.

    This is :func:`waspas` at :math:`\\lambda = 0`, and is implemented as that
    call so the two cannot drift apart. It is exposed separately because the
    method has its own name and literature, and nobody searching for WPM would
    think to look for it inside a WASPAS parameter.
    """
    return waspas(data, weights, directions, lambda_=0.0)


class Waspas(Method):
    """WASPAS as a registered method.

    Accepts ``lambda_`` through ``rank(..., lambda_=...)``: the weight given to
    the sum against the product.
    """

    name = "waspas"
    summary = "Blend of the weighted sum and weighted product models."
    citation = "Zavadskas, Turskis, Antucheviciene and Zakarevicius (2012)"
    wants = Wants.RAW
    requires_positive = True

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return waspas(ctx.data, ctx.weights, ctx.directions, **ctx.opts)


class Wpm(Method):
    """The weighted product model as a registered method.

    Takes no ``lambda_``: fixing it at zero is what distinguishes this from
    :class:`Waspas`.
    """

    name = "wpm"
    summary = "Weighted product model; a poor criterion cannot be paid for."
    citation = "Bridgman (1922); Miller and Starr (1969)"
    wants = Wants.RAW
    requires_positive = True

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return wpm(ctx.data, ctx.weights, ctx.directions)
