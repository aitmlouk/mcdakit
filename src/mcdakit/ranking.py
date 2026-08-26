"""The scoring entry points: :func:`rank` and :func:`compare_methods`."""

from __future__ import annotations

from collections.abc import Sequence

from .methods import (
    electre,
    promethee,
    saw,
    simple_scoring,
    spotis,
    topsis,
    vikor,
    weighted_scoring,
)
from .orientation import orient
from .types import Criterion, Decision, McdaError, Result

#: Every method name accepted by :func:`rank`, in a sensible reading order.
METHODS = (
    "simple_scoring",
    "weighted_scoring",
    "saw",
    "topsis",
    "vikor",
    "electre",
    "promethee",
    "spotis",
)

_ORIENTED = {
    "simple_scoring": simple_scoring,
    "weighted_scoring": weighted_scoring,
    "saw": saw,
    "topsis": topsis,
    "vikor": vikor,
    "electre": electre,
}


def as_decision(
    matrix,
    criteria: Sequence[Criterion] | None = None,
    labels: Sequence[str] | None = None,
) -> Decision:
    """Accept either a ``Decision`` or the pieces to build one."""
    if isinstance(matrix, Decision):
        if criteria is not None or labels is not None:
            raise McdaError(
                "Pass either a Decision or (matrix, criteria, labels), not "
                "both."
            )
        return matrix
    if criteria is None:
        raise McdaError("criteria are required when matrix is not a Decision.")
    return Decision(matrix, criteria, labels)


def score(decision: Decision, method: str = "weighted_scoring", **kwargs):
    """Raw scores for one method, without wrapping them in a :class:`Result`.

    Returns ``(scores, reversal_free, messages)``. Mostly useful internally —
    the sensitivity analysis re-scores the same problem hundreds of times and
    has no use for the surrounding object.
    """
    if method not in METHODS:
        raise McdaError(
            f"Unknown method {method!r}. Available: {', '.join(METHODS)}."
        )

    weights = decision.weights

    if method == "spotis":
        return spotis(
            decision.matrix,
            weights,
            decision.directions,
            [c.bounds for c in decision.criteria],
            **kwargs,
        )

    data = orient(decision.matrix, decision.directions)

    if method == "promethee":
        shapes = _shapes(decision)
        return promethee(data, weights, shapes, **kwargs), False, ()

    return _ORIENTED[method](data, weights, **kwargs), False, ()


def _shapes(decision: Decision):
    """``(shape, q, p, s)`` per criterion, or ``None`` if all are the default.

    Returning ``None`` when nothing was configured keeps PROMETHEE on its
    simple path, so a problem that never touched preference functions scores
    bit-identically to one built before they existed.
    """
    shapes = [
        (c.preference_shape, c.q, c.p, c.s) for c in decision.criteria
    ]
    if all(shape == "usual" for shape, _q, _p, _s in shapes):
        return None
    return shapes


def rank(
    matrix,
    criteria: Sequence[Criterion] | None = None,
    method: str = "weighted_scoring",
    labels: Sequence[str] | None = None,
    **kwargs,
) -> Result:
    """Rank the options in a decision matrix.

    Parameters
    ----------
    matrix:
        Options (rows) by criteria (columns), holding raw figures as measured.
        Cost criteria are handled by the ``direction`` on each
        :class:`~mcdakit.types.Criterion`; do not pre-invert them. A
        :class:`~mcdakit.types.Decision` may be passed instead, in which case
        ``criteria`` and ``labels`` must be omitted.
    method:
        One of :data:`METHODS`. The default, ``weighted_scoring``, is the
        least surprising; ``spotis`` is the one whose ranking cannot reverse.

    Returns
    -------
    Result
        With ``.winner``, ``.ranking``, ``.scores``, and — for SPOTIS —
        ``.reversal_free`` and any ``.warnings``.

    Examples
    --------
    >>> from mcdakit import Criterion, rank
    >>> criteria = [
    ...     Criterion("Price", weight=0.6, direction="cost"),
    ...     Criterion("Quality", weight=0.4),
    ... ]
    >>> result = rank([[3.0, 8], [2.0, 6]], criteria, labels=["A", "B"])
    >>> result.winner
    'B'
    """
    decision = as_decision(matrix, criteria, labels)
    scores, reversal_free, messages = score(decision, method, **kwargs)
    return Result(
        decision=decision,
        method=method,
        scores=scores,
        reversal_free=reversal_free,
        warnings=messages,
    )


def compare_methods(
    matrix,
    criteria: Sequence[Criterion] | None = None,
    labels: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
    **kwargs,
) -> dict:
    """Rank the same problem under every method.

    Returns ``{method_name: Result}``. When the methods disagree, that
    disagreement *is* the finding: it says the options are close enough that
    modelling choices decide the outcome, and collapsing it into a single
    number would hide the most useful thing on the table.

    See :func:`agreement` for a one-line summary of the same output.
    """
    decision = as_decision(matrix, criteria, labels)
    chosen = tuple(methods) if methods else METHODS
    for name in chosen:
        if name not in METHODS:
            raise McdaError(
                f"Unknown method {name!r}. Available: {', '.join(METHODS)}."
            )
    return {name: rank(decision, method=name, **kwargs) for name in chosen}


def agreement(results: dict) -> dict:
    """How much a :func:`compare_methods` mapping actually agrees.

    ``winners`` counts how many methods picked each option;
    ``consensus`` is the option most methods picked; ``unanimous`` says
    whether every method agreed on the whole order, not merely the winner.
    """
    winners = {}
    for result in results.values():
        winners[result.winner] = winners.get(result.winner, 0) + 1

    orders = {tuple(result.order) for result in results.values()}
    consensus = max(winners, key=lambda name: winners[name]) if winners else None

    return {
        "winners": winners,
        "consensus": consensus,
        "votes": winners.get(consensus, 0),
        "of": len(results),
        "unanimous": len(orders) == 1,
    }


def reversal_check(
    matrix,
    criteria: Sequence[Criterion] | None = None,
    method: str = "weighted_scoring",
    labels: Sequence[str] | None = None,
    **kwargs,
) -> dict:
    """Does dropping a losing option change the order of the survivors?

    Removes each non-winning option in turn, re-ranks what is left, and
    compares the survivors' order with their order in the full problem. This
    is the question behind the whole package: a ranking that changes when an
    option nobody chose is taken off the table was never really a ranking of
    the others.

    Returns ``{"reversed": bool, "cases": [...]}``, where each case names the
    option removed, the order before, and the order after.
    """
    decision = as_decision(matrix, criteria, labels)
    baseline = rank(decision, method=method, **kwargs)

    cases = []
    for index, label in enumerate(decision.labels):
        if index == baseline.winner_index:
            continue
        reduced = decision.without(index)
        after = rank(reduced, method=method, **kwargs)
        before = [name for name in baseline.order if name != label]
        if before != after.order:
            cases.append(
                {
                    "removed": label,
                    "before": before,
                    "after": after.order,
                }
            )

    return {"reversed": bool(cases), "cases": cases}
