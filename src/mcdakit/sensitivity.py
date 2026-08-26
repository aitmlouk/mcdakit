"""How far the weights can move before the winner changes.

A ranking is only as good as its weights, and weights are guessed. The
question that follows immediately — and that most MCDA tools never answer — is
how far someone would have to disagree with those guesses before the answer
changes. A winner that survives a 30% shift in every weight is a different
finding from one that flips at 4%, even when both are printed as "first".
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .ranking import score
from .types import Decision, McdaError, Result

#: Below this tolerance the decision is fragile enough to say so out loud.
FRAGILE_THRESHOLD = 0.10
#: At or above this, the winner is robust to any plausible re-weighting.
ROBUST_THRESHOLD = 0.25

#: Bisection steps. 40 halvings take the interval to roughly 1e-12 of its
#: original width, far below any weight difference that could matter.
BISECTION_STEPS = 40


def _winner_index(decision: Decision, weights: np.ndarray, method: str) -> int | None:
    """Index of the top option under these weights."""
    reweighted = _with_weights(decision, weights)
    with warnings.catch_warnings():
        # A bisection re-scores the same problem hundreds of times. Any
        # warning the method emits — SPOTIS's missing-bounds notice, or
        # anything a third-party method raises — is about the problem, not
        # about this particular trial weighting, and the caller has already
        # heard it once from the original ranking. Repeating it hundreds of
        # times would bury it. Suppressing here rather than passing a
        # method-specific flag keeps this working for methods we have never
        # seen.
        warnings.simplefilter("ignore")
        scores, _free, _messages = score(reweighted, method)
    if scores is None or len(scores) == 0:
        return None
    return int(np.argmax(scores))


def _with_weights(decision: Decision, weights) -> Decision:
    """The same decision with different criterion weights."""
    criteria = [
        dataclasses.replace(criterion, weight=float(weight))
        for criterion, weight in zip(decision.criteria, weights)
    ]
    return Decision(decision.matrix, criteria, decision.labels)


def _flip_point(
    decision: Decision,
    weights: np.ndarray,
    index: int,
    baseline: int,
    direction: int,
    method: str,
) -> float | None:
    """Smallest change to ``weights[index]`` that unseats ``baseline``.

    ``direction`` is ``+1`` to raise the weight, ``-1`` to lower it. Returns
    the change as a share of the total weight, or ``None`` if the winner
    survives across the whole searchable range.

    Bisection on the real scoring function, deliberately. The methods are not
    all linear in the weights — TOPSIS and VIKOR normalise, ELECTRE and
    PROMETHEE are step functions — so an analytic shortcut would be right for
    the additive methods and quietly wrong for the rest.
    """
    total = float(np.sum(weights)) or 1.0
    original = float(weights[index])

    # The furthest the weight can move: down to zero, or up until it dwarfs
    # everything else. Four times the budget is well past the point where one
    # criterion decides the outcome alone.
    limit = original if direction < 0 else max(total * 4.0, original * 4.0) - original
    if limit <= 0:
        return None

    def winner_at(delta: float) -> int | None:
        trial = weights.astype(float).copy()
        trial[index] = max(0.0, original + direction * delta)
        if np.sum(trial) <= 0:
            return baseline
        return _winner_index(decision, trial, method)

    if winner_at(limit) == baseline:
        return None

    low, high = 0.0, limit
    for _ in range(BISECTION_STEPS):
        mid = (low + high) / 2.0
        if winner_at(mid) == baseline:
            low = mid
        else:
            high = mid
    return high / total


def sensitivity(result: Result) -> dict:
    """Per-criterion weight tolerance for a ranking.

    For each criterion, how far its weight must move — up or down, as a share
    of the total weight — before a different option wins, and which option
    takes over. The smallest such move across all criteria is the decision's
    overall tolerance.

    Returns
    -------
    dict
        ``overall``
            Smallest tolerance across the criteria, or ``None`` if no single
            weight change can unseat the winner.
        ``level``
            ``"fragile"`` under 10%, ``"moderate"`` under 25%, ``"robust"``
            at or above, ``"immovable"`` when nothing flips it.
        ``weakest``
            Name of the criterion carrying that smallest tolerance.
        ``rows``
            One dict per criterion: ``name``, ``weight``, ``increase``,
            ``decrease``, ``tolerance``, ``direction``, ``flips_to``.

    Which *way* the weight has to move matters as much as how far: "Quality
    can move by 7.5%" is unusable without knowing that means down.

    Only one weight moves at a time. A simultaneous shift in several could
    unseat the winner sooner, so this is an upper bound on stability, not a
    guarantee — a "robust" result is robust against single-criterion
    disagreement, which is the form disagreement usually takes in practice.

    Examples
    --------
    >>> from mcdakit import Criterion, rank, sensitivity
    >>> criteria = [Criterion("Price", 0.6, "cost"), Criterion("Quality", 0.4)]
    >>> result = rank([[3.0, 8], [2.0, 6]], criteria, labels=["A", "B"])
    >>> sensitivity(result)["level"] in {
    ...     "fragile", "moderate", "robust", "immovable"}
    True
    """
    if not isinstance(result, Result):
        raise McdaError(
            f"sensitivity() takes a Result from rank(), got {type(result).__name__}."
        )

    decision = result.decision
    method = result.method

    if decision.n_options < 2:
        raise McdaError(
            "Stability only means something when there is more than one "
            "option to choose between."
        )

    weights = decision.weights
    baseline = _winner_index(decision, weights, method)
    if baseline is None:
        raise McdaError("This decision cannot be scored.")

    total = float(np.sum(weights)) or 1.0
    rows = []

    for index, criterion in enumerate(decision.criteria):
        up = _flip_point(decision, weights, index, baseline, +1, method)
        down = _flip_point(decision, weights, index, baseline, -1, method)

        candidates = [c for c in (up, down) if c is not None]
        smallest = min(candidates) if candidates else None

        if smallest is None:
            movement = ""
            flips_to = ""
        else:
            sign = +1 if (up is not None and smallest == up) else -1
            movement = "up" if sign > 0 else "down"
            # Step just past the flip point to read off who takes over.
            nudge = weights.astype(float).copy()
            nudge[index] = max(
                0.0, float(weights[index]) + sign * smallest * total * 1.001
            )
            flipped = _winner_index(decision, nudge, method)
            flips_to = decision.labels[flipped] if flipped is not None else ""

        rows.append(
            {
                "name": criterion.name,
                "weight": criterion.weight,
                "increase": up,
                "decrease": down,
                "tolerance": smallest,
                "direction": movement,
                "flips_to": flips_to,
            }
        )

    tolerances = [r["tolerance"] for r in rows if r["tolerance"] is not None]
    overall = min(tolerances) if tolerances else None

    if overall is None:
        level = "immovable"
    elif overall < FRAGILE_THRESHOLD:
        level = "fragile"
    elif overall < ROBUST_THRESHOLD:
        level = "moderate"
    else:
        level = "robust"

    weakest = min(
        (r for r in rows if r["tolerance"] is not None),
        key=lambda r: r["tolerance"],
        default=None,
    )

    return {
        "method": method,
        "winner": decision.labels[baseline],
        "overall": overall,
        "level": level,
        "weakest": weakest["name"] if weakest else "",
        "weakest_direction": weakest["direction"] if weakest else "",
        "weakest_flips_to": weakest["flips_to"] if weakest else "",
        "rows": rows,
    }
