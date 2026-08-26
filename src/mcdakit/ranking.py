"""The scoring entry points: :func:`rank` and :func:`compare_methods`."""

from __future__ import annotations

from collections.abc import Sequence

from .methods import get as get_method
from .methods import names as method_names
from .methods.base import ScoringContext, Wants, _as_method_result
from .orientation import orient
from .types import Criterion, Decision, McdaError, Result


class _MethodNames(Sequence):
    """A live view of the registered method names.

    ``METHODS`` used to be a frozen tuple. Now that methods can be registered
    at any time — by a plugin, or by the user mid-session — a snapshot taken at
    import would go stale. This reads the registry on every access, so
    ``"my_method" in METHODS`` is true the moment it is registered, while
    still supporting everything a tuple supported.
    """

    def __getitem__(self, index):
        return method_names()[index]

    def __len__(self) -> int:
        return len(method_names())

    def __iter__(self):
        return iter(method_names())

    def __contains__(self, value) -> bool:
        return value in method_names()

    def __eq__(self, other) -> bool:
        return tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))

    def __repr__(self) -> str:
        return repr(method_names())


#: Every registered method name — built-in and plugin alike. A live view of
#: the registry, not a snapshot.
METHODS = _MethodNames()


def as_decision(
    matrix,
    criteria: Sequence[Criterion] | None = None,
    labels: Sequence[str] | None = None,
) -> Decision:
    """Accept either a ``Decision`` or the pieces to build one."""
    if isinstance(matrix, Decision):
        if criteria is not None or labels is not None:
            raise McdaError(
                "Pass either a Decision or (matrix, criteria, labels), not both."
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

    Dispatch goes through the registry, so a method registered by a third-party
    package is reached the same way a built-in is. Nothing here branches on a
    method's name; what a method needs, it declares.
    """
    implementation = get_method(method)
    implementation.validate(decision)

    if implementation.wants is Wants.RAW:
        data = decision.matrix
    else:
        data = orient(decision.matrix, decision.directions)

    ctx = ScoringContext(
        data=data,
        weights=decision.weights,
        decision=decision,
        options=kwargs,
    )
    result = _as_method_result(implementation.score(ctx))

    if implementation.wants is not Wants.RAW and ctx.read_directions:
        # The matrix arrived with cost columns already mirrored, and the method
        # then consulted directions itself — so it will flip them a second
        # time and rank costs exactly backwards. Silent, and catastrophic: the
        # cheapest option comes last. A method that handles direction itself
        # must ask for the raw matrix.
        raise McdaError(
            f"Method {method!r} read ctx.directions but declares "
            f"wants = Wants.ORIENTED, so it received a matrix whose cost "
            f"columns were already flipped. Handling direction again inverts "
            f"them. Set `wants = Wants.RAW` on "
            f"{type(implementation).__name__} to receive the figures as "
            f"measured, or stop reading ctx.directions and let the library "
            f"orient them."
        )

    if kwargs and not ctx.consumed:
        # The method never looked at ctx.options, so these went nowhere.
        # Silently ignoring them is how `rank(..., v=0.5)` on a method with no
        # `v` came to be a no-op that looked like it worked.
        raise McdaError(
            f"Method {method!r} takes no keyword arguments, but got "
            f"{', '.join(sorted(kwargs))}. Check the spelling, or see "
            f"{type(implementation).__name__}.score for what it accepts."
        )
    return result.scores, result.reversal_free, result.warnings


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
        Any registered method name; see :func:`mcdakit.methods.names`.
        The default, ``weighted_scoring``, is the
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
    chosen = tuple(methods) if methods else method_names()
    for name in chosen:
        get_method(name)  # raises McdaError, listing what is available
    return {name: rank(decision, method=name, **kwargs) for name in chosen}


def agreement(results: dict) -> dict:
    """How much a :func:`compare_methods` mapping actually agrees.

    ``winners`` counts how many methods picked each option;
    ``consensus`` is the option most methods picked; ``unanimous`` says
    whether every method agreed on the whole order, not merely the winner.
    """
    winners: dict = {}
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
