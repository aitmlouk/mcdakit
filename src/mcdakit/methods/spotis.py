"""SPOTIS — Stable Preference Ordering Towards Ideal Solution.

Dezert, Tchamova, Han and Tacnet, *The SPOTIS Rank Reversal Free Method for
Multi-Criteria Decision-Making Support*, FUSION 2020.

The one method here that cannot reverse rank. Every other method in this
package measures options against each other in some way — TOPSIS against the
best and worst present, VIKOR against the observed range, weighted scoring
against the observed span — so removing an option moves the yardstick and can
reorder the options that remain. SPOTIS measures each option against a fixed
ideal point derived from bounds the analyst sets in advance. Nothing an option
does can move another option's score, so no addition or removal can change the
order of the rest.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ..normalization import normalize_weights
from .base import Method, MethodResult, ScoringContext, Wants


class BoundsWarning(UserWarning):
    """A SPOTIS ranking fell back to observed data instead of fixed bounds.

    Its own category so callers can turn it into an error with
    ``warnings.simplefilter("error", BoundsWarning)`` — worth doing in any
    pipeline where the reversal-freedom guarantee is the reason SPOTIS was
    chosen.
    """


def spotis(
    data: np.ndarray,
    weights: np.ndarray,
    directions: Sequence[str],
    bounds: Sequence | None = None,
    warn: bool = True,
):
    """Rank-reversal-free scoring, returned **negated** so higher is better.

    .. math::

        d_{ij} = \\frac{|S_{ij} - S^*_j|}{|S_j^{max} - S_j^{min}|}
        \\qquad
        d(A_i, S^*) = \\sum_j w_j \\, d_{ij}

    ``S*`` is the ideal point: the best value each criterion *could* take,
    read off the bounds rather than off the options — ``hi`` for a benefit
    criterion, ``lo`` for a cost. The denominator is the bounded span, also
    fixed. So ``d_ij`` depends only on option ``i``, and every other option is
    irrelevant to it.

    SPOTIS's native distance is smallest-is-best. This function returns
    ``-d`` so that, like every other method here, larger scores rank higher.
    Scores are therefore in ``[-1, 0]``, where ``0`` is an option sitting
    exactly on the ideal point. **Read the sign before comparing these numbers
    with published SPOTIS tables, which report the positive distance.**

    Parameters
    ----------
    data:
        Raw, **un-oriented** matrix. Unlike the other methods, SPOTIS handles
        direction itself through the choice of ideal point, and orienting the
        matrix first would break the correspondence with the bounds.
    bounds:
        One ``(lo, hi)`` pair per criterion, or ``None`` for a criterion whose
        range is not known.

    Returns
    -------
    (scores, reversal_free, messages)
        ``reversal_free`` is ``False`` when any criterion had to fall back to
        observed min/max, along with a message naming which. **In that case the
        method is not reversal-free** — the fallback bounds move when the
        option set moves, which is exactly the mechanism SPOTIS exists to
        avoid. Silently returning a number that looks authoritative would be
        the worst failure this package could have, so the shortfall is both
        warned about and reported in the return value.
    """
    data = np.asarray(data, dtype=float)
    n_alt, n_crit = data.shape

    w = normalize_weights(weights)
    if w is None:
        return np.zeros(n_alt), True, ()

    if bounds is None:
        bounds = [None] * n_crit

    lo = np.empty(n_crit)
    hi = np.empty(n_crit)
    fell_back = []

    for j in range(n_crit):
        pair = bounds[j] if j < len(bounds) else None
        if pair is None:
            lo[j] = float(np.min(data[:, j]))
            hi[j] = float(np.max(data[:, j]))
            fell_back.append(j)
        else:
            lo[j], hi[j] = float(pair[0]), float(pair[1])

    span = hi - lo
    # A criterion whose bounded span is zero cannot discriminate; a flat
    # distance of zero is the honest answer, not a division by zero.
    span = np.where(span == 0, 1.0, span)

    ideal = np.where(np.asarray(directions) == "cost", lo, hi)

    distances = np.abs(data - ideal) / span
    total = distances @ w

    messages: tuple = ()
    reversal_free = not fell_back
    if fell_back:
        names = ", ".join(f"criterion {j + 1}" for j in fell_back)
        message = (
            f"SPOTIS fell back to observed min/max for {names} because no "
            f"bounds were given. The result is NOT rank-reversal-free: those "
            f"bounds move when the option set moves. Set bounds=(lo, hi) on "
            f"every criterion to get the guarantee."
        )
        messages = (message,)
        if warn:
            warnings.warn(message, BoundsWarning, stacklevel=2)

    return (-total).astype(float), reversal_free, messages


class Spotis(Method):
    """SPOTIS as a registered method.

    The one built-in that declares :attr:`Wants.RAW`: it must see the figures
    as measured, because it resolves criterion direction itself against the
    fixed bounds rather than against the other options.

    :attr:`reversal_free` is ``True`` as a statement about the method, but each
    call reports its own verdict — a problem with missing bounds comes back
    ``reversal_free=False`` with a warning, because the guarantee is a property
    of the inputs as much as of the algorithm.
    """

    name = "spotis"
    summary = "Rank-reversal-free distance to a fixed ideal point."
    citation = "Dezert, Tchamova, Han and Tacnet (2020)"
    wants = Wants.RAW
    reversal_free = True

    def score(self, ctx: ScoringContext) -> MethodResult:
        scores, reversal_free, messages = spotis(
            ctx.data,
            ctx.weights,
            ctx.directions,
            ctx.bounds,
            **ctx.opts,
        )
        return MethodResult(
            scores=scores, reversal_free=reversal_free, warnings=messages
        )
