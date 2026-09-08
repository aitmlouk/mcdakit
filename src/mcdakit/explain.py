"""Why did this option win?

A ranking is a conclusion, and a conclusion nobody can interrogate is
difficult to act on or to defend. Transparency is increasingly treated in the
decision-analysis literature as a precondition for trusting a computed result
rather than as a presentational extra.

This module answers two questions about a ranking the package has already
produced:

* **Which criteria carried the result?** Each criterion's contribution to an
  option's score, and how much of the score it accounts for.
* **Why did A beat B?** The margin between two options, decomposed by
  criterion, showing which criteria favoured the winner, which worked against
  it, and whether any single one was decisive.

Two ways of attributing a score are used, because only one of them is exact
and it does not apply everywhere:

``exact``
    The method is a weighted sum of normalised values, so an option's score
    *is* the sum of its per-criterion contributions. Nothing is approximated
    and the parts add up to the whole. Applies to ``weighted_scoring``,
    ``saw``, ``simple_scoring`` and ``spotis``.

``leave_one_out``
    The method is not additive --- TOPSIS is a ratio of distances, VIKOR
    takes a maximum, ELECTRE and PROMETHEE compare options pairwise --- so no
    exact decomposition exists. Each criterion's importance is instead
    measured by removing it and re-scoring: the contribution is how much the
    score changes in its absence. These values are attributions, not
    components, and do not sum to the score.

Which was used is reported on every result, because presenting an
approximation as though it were a decomposition would be the same
overstatement the rest of this package exists to avoid.

    >>> from mcdakit import Criterion, rank
    >>> from mcdakit.explain import explain
    >>> criteria = [Criterion("Price", 0.6, "cost"), Criterion("Quality", 0.4)]
    >>> result = rank([[3.0, 8.0], [2.0, 6.0]], criteria, labels=["A", "B"])
    >>> report = explain(result)
    >>> report.basis
    'exact'
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import McdaError, Result

__all__ = ["Contribution", "Explanation", "Margin", "compare", "explain"]

#: Methods whose score is exactly the sum of per-criterion contributions.
ADDITIVE = frozenset({"simple_scoring", "weighted_scoring", "saw", "spotis"})


@dataclass(frozen=True)
class Contribution:
    """One criterion's part in one option's score."""

    criterion: str
    value: float
    weight: float
    contribution: float
    share: float

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return (
            f"{self.criterion}: {self.contribution:+.4f} "
            f"({self.share:.0%} of the score)"
        )


@dataclass(frozen=True)
class Explanation:
    """Why one option scored what it did.

    Attributes
    ----------
    basis:
        ``"exact"`` when the contributions sum to the score, or
        ``"leave_one_out"`` when they are attributions obtained by removing
        each criterion in turn. The distinction is reported rather than
        smoothed over: only the first is a decomposition.
    dominant:
        The criterion contributing most. For a close decision this is the one
        worth checking first.
    """

    option: str
    method: str
    score: float
    basis: str
    contributions: tuple
    dominant: str

    def __str__(self) -> str:
        lines = [
            f"{self.option} scored {self.score:.4f} under {self.method}",
            f"  attribution: {self.basis}",
        ]
        for c in self.contributions:
            lines.append(f"  {c.criterion:<14}{c.contribution:+.4f}  ({c.share:>5.0%})")
        if self.basis == "leave_one_out":
            lines.append(
                "  (attributions from removing each criterion; they do not "
                "sum to the score)"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class Margin:
    """Why one option beat another, criterion by criterion."""

    winner: str
    loser: str
    method: str
    margin: float
    basis: str
    rows: tuple
    decisive: str | None = None

    def __str__(self) -> str:
        lines = [
            f"{self.winner} beats {self.loser} by {self.margin:.4f} "
            f"under {self.method}",
            f"  attribution: {self.basis}",
        ]
        for criterion, delta in self.rows:
            # A criterion the two options score alike separates them not at
            # all; saying it "favours" either would misread a tie.
            if delta == 0:
                side = "no difference"
            else:
                side = "favours " + (self.winner if delta > 0 else self.loser)
            lines.append(f"  {criterion:<14}{delta:+.4f}  {side}")
        if self.decisive:
            lines.append(
                f"  {self.decisive} is decisive: without it the result reverses"
            )
        return "\n".join(lines)


def _oriented_normalised(result: Result):
    """The matrix as the method saw it, plus the weights it applied."""
    from .methods import get as get_method
    from .methods.base import Wants
    from .normalization import normalize
    from .orientation import orient

    decision = result.decision
    method = get_method(result.method)

    if method.wants is Wants.RAW:
        data = np.asarray(decision.matrix, dtype=float)
    else:
        data = orient(decision.matrix, decision.directions)

    if method.normalization is not None:
        data = normalize(data, method.normalization)

    return data, decision.normalized_weights


def _exact_contributions(result: Result, index: int):
    """Per-criterion contributions for an additive method."""
    data, weights = _oriented_normalised(result)

    if result.method == "spotis":
        # SPOTIS is additive in the distance, and the score is its negation:
        # each criterion contributes -w * d, so the parts still sum to the
        # score and a larger (less negative) part is a better one.
        from .methods.spotis import spotis

        decision = result.decision
        bounds = [c.bounds for c in decision.criteria]
        lo = np.array(
            [
                b[0] if b else float(np.min(decision.matrix[:, j]))
                for j, b in enumerate(bounds)
            ]
        )
        hi = np.array(
            [
                b[1] if b else float(np.max(decision.matrix[:, j]))
                for j, b in enumerate(bounds)
            ]
        )
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        ideal = np.where(np.asarray(decision.directions) == "cost", lo, hi)
        del spotis
        return -weights * np.abs(decision.matrix[index] - ideal) / span

    if result.method == "simple_scoring":
        # Ignores weights by design, so each criterion contributes its own
        # oriented value.
        return data[index]

    return weights * data[index]


def _leave_one_out_contributions(result: Result, index: int):
    """Attribution by removal, for a method that does not decompose.

    A criterion's importance is how much the option's score changes when that
    criterion is dropped and the problem re-scored. Dropping a criterion means
    setting its weight to zero, which is what "this criterion did not matter"
    means in the model, rather than deleting a column and changing the shape
    of the problem.
    """
    import dataclasses

    from .ranking import score
    from .types import Decision

    decision = result.decision
    baseline = float(result.scores[index])
    weights = decision.weights

    contributions = []
    for j in range(decision.n_criteria):
        muted = weights.astype(float).copy()
        muted[j] = 0.0
        if muted.sum() <= 0:
            # A single-criterion problem: everything rests on it.
            contributions.append(baseline)
            continue
        criteria = [
            dataclasses.replace(c, weight=float(w))
            for c, w in zip(decision.criteria, muted)
        ]
        without = Decision(decision.matrix, criteria, decision.labels)
        scores, _free, _messages = score(without, result.method)
        contributions.append(baseline - float(scores[index]))
    return np.array(contributions)


def explain(result: Result, option: str | None = None) -> Explanation:
    """Explain how one option's score arose.

    Parameters
    ----------
    result:
        A :class:`~mcdakit.types.Result` from :func:`~mcdakit.rank`.
    option:
        Which option to explain. Defaults to the winner, since that is the one
        a reader will question first.

    Returns
    -------
    Explanation
        Per-criterion contributions, and whether they are an exact
        decomposition or attributions obtained by leaving each criterion out.
    """
    if not isinstance(result, Result):
        raise McdaError(
            f"explain() takes a Result from rank(), got {type(result).__name__}."
        )

    decision = result.decision
    label = option if option is not None else result.winner
    try:
        index = decision.labels.index(label)
    except ValueError as exc:
        raise McdaError(f"No option named {label!r}.") from exc

    if result.method in ADDITIVE:
        basis = "exact"
        raw = _exact_contributions(result, index)
    else:
        basis = "leave_one_out"
        raw = _leave_one_out_contributions(result, index)

    total = float(np.sum(np.abs(raw))) or 1.0
    contributions = tuple(
        Contribution(
            criterion=criterion.name,
            value=float(decision.matrix[index, j]),
            weight=float(criterion.weight),
            contribution=float(raw[j]),
            share=float(abs(raw[j]) / total),
        )
        for j, criterion in enumerate(decision.criteria)
    )
    ordered = tuple(sorted(contributions, key=lambda c: -abs(c.contribution)))

    return Explanation(
        option=label,
        method=result.method,
        score=float(result.scores[index]),
        basis=basis,
        contributions=ordered,
        dominant=ordered[0].criterion,
    )


def compare(result: Result, winner: str, loser: str) -> Margin:
    """Explain why one option outranked another.

    The margin between two options is decomposed by criterion, so a reader can
    see which criteria favoured the winner and which worked against it. A
    criterion is reported as *decisive* when reversing its contribution alone
    would reverse the outcome --- the case where a single disputed judgement
    determines the recommendation, and the one most worth knowing about.
    """
    first = explain(result, winner)
    second = explain(result, loser)

    if first.score < second.score:
        raise McdaError(
            f"{winner!r} did not outrank {loser!r} ({first.score:.4f} vs "
            f"{second.score:.4f}). Pass the higher-scoring option first."
        )

    by_name = {c.criterion: c.contribution for c in second.contributions}
    rows = tuple(
        (c.criterion, c.contribution - by_name[c.criterion])
        for c in sorted(first.contributions, key=lambda c: c.criterion)
    )

    margin = first.score - second.score
    decisive = None
    for criterion, delta in rows:
        # Removing this criterion's advantage would put the loser ahead.
        if delta > 0 and delta > margin:
            decisive = criterion
            break

    return Margin(
        winner=winner,
        loser=loser,
        method=result.method,
        margin=float(margin),
        basis=first.basis,
        rows=tuple(sorted(rows, key=lambda r: -abs(r[1]))),
        decisive=decisive,
    )
