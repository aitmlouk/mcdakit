"""Decisions made by several people.

Two people scoring the same options rarely agree, and the interesting question
is not only "what did the group decide?" but "how much did they disagree, and
does the answer survive that disagreement?" A consensus ranking that hides a
6-2 split is the same failure as a point estimate that hides its error bars.

There are two honest ways to combine several stakeholders, and they answer
different questions:

**Aggregate the scores, then rank once** (:func:`aggregate_scores`). Treats the
group as one better-informed judge: average what each person thought each
option was worth, then run a method over the averages. Right when the
participants are measuring the same thing and their differences are noise.

**Rank separately, then combine the rankings** (:func:`aggregate_rankings`).
Treats each participant as a voter with their own legitimate view. Right when
the differences are real preferences rather than measurement error, and when
you do not want one person's outlying score to drag the average.

They can disagree, and when they do that is a finding about the group, not a
defect. :func:`group_rank` reports both by default.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ranking import rank
from .types import Criterion, Decision, McdaError

__all__ = [
    "Participant",
    "aggregate_rankings",
    "aggregate_scores",
    "disagreement",
    "group_rank",
]


class Participant:
    """One stakeholder's view of the same decision.

    Parameters
    ----------
    name:
        Who this is. Used in the disagreement report.
    matrix:
        Their scores. Must cover the same options and criteria as everyone
        else, in the same order — see :func:`group_rank` for why.
    weight:
        How much this participant counts. Equal by default. A chair with a
        casting vote, or a domain expert weighted above a generalist, is
        expressed here rather than by duplicating their row.
    """

    def __init__(self, name: str, matrix, weight: float = 1.0):
        if not str(name).strip():
            raise McdaError("Every participant needs a name.")
        weight = float(weight)
        if weight < 0 or not np.isfinite(weight):
            raise McdaError(
                f"Participant {name!r}: weight must be a non-negative finite "
                f"number, got {weight}."
            )
        self.name = str(name)
        self.matrix = np.asarray(matrix, dtype=float)
        self.weight = weight

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Participant {self.name!r} weight={self.weight}>"


def _check(participants: Sequence[Participant], criteria: Sequence[Criterion]):
    """Refuse a group whose members are not judging the same problem.

    The mistake this prevents: participants scoring different option sets get
    averaged over different numbers of people, so an option only one person
    rated is compared against options everyone rated. The result looks
    authoritative and is computed from uneven evidence.
    """
    if not participants:
        raise McdaError("A group decision needs at least one participant.")
    if len(participants) < 2:
        raise McdaError(
            "A group decision needs at least two participants; with one there "
            "is nothing to combine. Use rank() instead."
        )

    shapes = {p.matrix.shape for p in participants}
    if len(shapes) > 1:
        detail = ", ".join(f"{p.name} {p.matrix.shape}" for p in participants)
        raise McdaError(
            f"Participants are not scoring the same problem: {detail}. Every "
            f"matrix must have the same options (rows) and criteria "
            f"(columns), in the same order, or their scores would be averaged "
            f"over different people."
        )

    rows, cols = participants[0].matrix.shape
    if cols != len(criteria):
        raise McdaError(
            f"Each participant scored {cols} criteria but {len(criteria)} were given."
        )

    total = sum(p.weight for p in participants)
    if total <= 0:
        raise McdaError(
            "Participant weights are all zero, so nobody influences the result."
        )
    return rows, cols


def _participant_weights(participants: Sequence[Participant]) -> np.ndarray:
    weights = np.array([p.weight for p in participants], dtype=float)
    return weights / weights.sum()


def aggregate_scores(
    participants: Sequence[Participant],
    criteria: Sequence[Criterion],
    labels: Sequence[str] | None = None,
) -> Decision:
    """Average the participants' scores into one decision matrix.

    The weighted mean, cell by cell. Treats the group as a single
    better-informed judge, which is the right model when the participants are
    estimating the same underlying quantity and differ only by noise.

    It is the wrong model when they hold genuinely different preferences: an
    average of 9 and 2 is 5.5, a value neither person would recognise, and
    nothing in the resulting ranking records that they disagreed. Pair it with
    :func:`disagreement`, which is what :func:`group_rank` does.
    """
    _check(participants, criteria)
    shares = _participant_weights(participants)
    stacked = np.stack([p.matrix for p in participants])
    averaged = np.tensordot(shares, stacked, axes=(0, 0))
    return Decision(averaged, criteria, labels)


def aggregate_rankings(
    participants: Sequence[Participant],
    criteria: Sequence[Criterion],
    method: str = "weighted_scoring",
    labels: Sequence[str] | None = None,
    **kwargs,
) -> dict:
    """Rank for each participant separately, then combine the rankings.

    Borda count over each participant's ordering: an option placed first by
    someone earns ``n - 1`` points from them, second earns ``n - 2``, and so
    on, weighted by that participant's weight. Ties share the average of the
    positions they span, so a tie cannot invent a preference nobody expressed.

    Unlike :func:`aggregate_scores` this is insensitive to *how far apart* the
    scores were — only to the order each person put them in. That is a
    weakness when the magnitudes are meaningful and a strength when one
    participant's outlying score would otherwise drag the mean.

    Returns ``{"scores", "order", "per_participant"}``, where
    ``per_participant`` maps each name to their own :class:`~mcdakit.Result`.
    """
    n_options, _ = _check(participants, criteria)
    shares = _participant_weights(participants)

    points = np.zeros(n_options)
    per_participant = {}

    for participant, share in zip(participants, shares):
        result = rank(
            participant.matrix,
            criteria,
            method=method,
            labels=labels,
            **kwargs,
        )
        per_participant[participant.name] = result
        # Borda points from competition ranks, so tied options score equally
        # rather than being separated by the order they happen to be listed.
        points += share * (n_options - result.ranks)

    order = sorted(range(n_options), key=lambda i: -points[i])
    names = tuple(
        labels if labels is not None else [f"Option {i + 1}" for i in range(n_options)]
    )
    return {
        "scores": points,
        "order": [names[i] for i in order],
        "per_participant": per_participant,
    }


def disagreement(
    participants: Sequence[Participant],
    criteria: Sequence[Criterion],
    labels: Sequence[str] | None = None,
) -> dict:
    """How far apart the participants were, per option and per criterion.

    Reported as the standard deviation of the scores each cell received,
    unweighted — the question is how much the people differed, not how much
    the chair's opinion outweighs the rest.

    A mean of 5.5 that everybody agreed on and one that averages 9 and 2 are
    the same number and completely different findings. This is what tells them
    apart.

    Returns ``overall``, ``by_option``, ``by_criterion``, and ``most_contested``
    naming the option and criterion with the widest spread.
    """
    _check(participants, criteria)
    stacked = np.stack([p.matrix for p in participants])
    spread = stacked.std(axis=0)

    names = (
        list(labels)
        if labels is not None
        else [f"Option {i + 1}" for i in range(spread.shape[0])]
    )
    criterion_names = [c.name for c in criteria]

    by_option = dict(zip(names, spread.mean(axis=1)))
    by_criterion = dict(zip(criterion_names, spread.mean(axis=0)))

    return {
        "overall": float(spread.mean()),
        "by_option": {k: float(v) for k, v in by_option.items()},
        "by_criterion": {k: float(v) for k, v in by_criterion.items()},
        "most_contested_option": max(by_option, key=lambda k: by_option[k]),
        "most_contested_criterion": max(by_criterion, key=lambda k: by_criterion[k]),
        "spread": spread,
    }


def group_rank(
    participants: Sequence[Participant],
    criteria: Sequence[Criterion],
    method: str = "weighted_scoring",
    labels: Sequence[str] | None = None,
    **kwargs,
) -> dict:
    """Rank a decision made by several people, and say how much they agreed.

    Runs both aggregations — averaging the scores, and combining each
    participant's ranking — because they answer different questions and can
    disagree. When they do, that disagreement is a finding about the group:
    it means the answer depends on whether you treat the participants as
    co-estimators of one truth or as voters with distinct views.

    Returns
    -------
    dict
        ``result``
            A :class:`~mcdakit.Result` from ranking the averaged scores. The
            headline answer.
        ``borda``
            The rank-aggregated view, from :func:`aggregate_rankings`.
        ``agree``
            Whether the two aggregations chose the same winner.
        ``disagreement``
            The spread report from :func:`disagreement`.
        ``per_participant``
            Each participant's own :class:`~mcdakit.Result`.
        ``unanimous``
            Whether every participant, ranking alone, picked the same winner.

    Examples
    --------
    >>> from mcdakit import Criterion
    >>> from mcdakit.group import Participant, group_rank
    >>> criteria = [Criterion("Price", 0.5, "cost"), Criterion("Quality", 0.5)]
    >>> alice = Participant("Alice", [[2.0, 8.0], [3.0, 6.0]])
    >>> bob = Participant("Bob", [[2.0, 5.0], [3.0, 9.0]])
    >>> out = group_rank([alice, bob], criteria, labels=["A", "B"])
    >>> out["agree"]
    True
    """
    decision = aggregate_scores(participants, criteria, labels)
    result = rank(decision, method=method, **kwargs)
    borda = aggregate_rankings(
        participants, criteria, method=method, labels=labels, **kwargs
    )

    winners = {r.winner for r in borda["per_participant"].values()}

    return {
        "result": result,
        "borda": borda,
        "agree": result.winner == borda["order"][0],
        "disagreement": disagreement(participants, criteria, labels),
        "per_participant": borda["per_participant"],
        "unanimous": len(winners) == 1,
    }
