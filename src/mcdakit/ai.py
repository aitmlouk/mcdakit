"""Language-model assistance, with every suggestion checked before it counts.

An LLM is useful in MCDA for the part that is genuinely hard to automate:
turning a described problem into a structured one. Naming plausible criteria,
proposing a direction for each, and drafting a first set of weights are tasks
where a model's breadth helps and where a wrong answer is cheap to spot.

It is *not* useful as an authority. A model that returns weights returns them
with the same confidence whether they are considered or invented, and nothing
in the output distinguishes the two. A package whose stated purpose is to
refuse answers it cannot stand behind must not make an exception here.

So this module inverts the usual arrangement. The model **proposes**; the
package **verifies**, and reports what the proposal is worth:

* every suggestion is parsed into the package's own validated types, so a
  malformed or out-of-range answer is rejected rather than absorbed;
* proposed weights are run through :func:`~mcdakit.sensitivity`, so the
  caller is told whether the recommendation would even survive the model
  being somewhat wrong;
* the same question is asked more than once and the answers compared, so a
  model that is guessing can be distinguished from one that is consistent;
* nothing is applied automatically. Every function returns a *proposal* the
  caller must choose to accept.

The provider is injected rather than imported. ``mcdakit`` acquires no LLM
dependency: the caller supplies any callable mapping a prompt to a string,
which may wrap any vendor's client, a local model, or a fixture in a test.

    >>> from mcdakit.ai import propose_criteria
    >>> reply = '''[{"name": "Price", "direction": "cost", "weight": 0.6},
    ...              {"name": "Quality", "direction": "benefit", "weight": 0.4}]'''
    >>> proposal = propose_criteria("choosing a supplier", ask=lambda p: reply)
    >>> [c.name for c in proposal.criteria]
    ['Price', 'Quality']
    >>> proposal.accepted
    False

Because the model is injected, this module is deterministic under test and
adds no network dependency to the package.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .types import Criterion, McdaError

__all__ = [
    "AiProposal",
    "WeightProposal",
    "critique_weights",
    "propose_criteria",
    "propose_weights",
]

#: A provider is any callable taking a prompt and returning the model's reply.
Ask = Callable[[str], str]


class AiError(McdaError):
    """A model's reply could not be turned into a valid proposal."""


@dataclass(frozen=True)
class AiProposal:
    """Criteria a model suggested, and what the package makes of them.

    Attributes
    ----------
    criteria:
        Validated :class:`~mcdakit.types.Criterion` objects. A suggestion that
        failed validation never reaches this list.
    rejected:
        Suggestions that were discarded, each with the reason. Reported rather
        than dropped: a model that proposed six criteria of which two were
        malformed is a different signal from one that proposed four.
    agreement:
        When the model was asked more than once, the share of criteria names
        common to every reply. Low agreement indicates the model is
        improvising, and is worth more than any single answer.
    accepted:
        Always ``False``. A proposal is a suggestion; applying it is the
        caller's decision, made by constructing a
        :class:`~mcdakit.types.Decision` from it.
    """

    criteria: tuple
    rejected: tuple = ()
    agreement: float | None = None
    raw: tuple = ()
    accepted: bool = False

    def __str__(self) -> str:
        lines = [f"{len(self.criteria)} criteria proposed:"]
        for criterion in self.criteria:
            lines.append(
                f"  {criterion.name} ({criterion.direction}, "
                f"weight {criterion.weight:.2f})"
            )
        for name, why in self.rejected:
            lines.append(f"  ! discarded {name!r}: {why}")
        if self.agreement is not None:
            lines.append(f"  agreement across replies: {self.agreement:.0%}")
        lines.append("  not applied — construct a Decision to accept")
        return "\n".join(lines)


@dataclass(frozen=True)
class WeightProposal:
    """Weights a model suggested, checked against the data.

    Attributes
    ----------
    weights:
        Normalised to sum to one.
    stability:
        The :func:`~mcdakit.sensitivity` report obtained *using* these weights.
        This is the point of the exercise: a model's weights may be plausible
        and still produce a recommendation that a small correction overturns,
        and the caller should know that before relying on them.
    disagrees_with:
        Objective schemes whose weights order the criteria differently. A
        model that ranks importance unlike every data-driven scheme is not
        necessarily wrong, but the divergence is worth surfacing.
    """

    weights: np.ndarray
    criteria: tuple
    stability: dict = field(default_factory=dict)
    disagrees_with: tuple = ()
    agreement: float | None = None
    accepted: bool = False


def _extract_json(reply: str):
    """Pull the first JSON array or object out of a model's reply.

    Models wrap JSON in prose and code fences however they were prompted not
    to, so the reply is searched rather than parsed whole.
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)```", reply, re.S)
    if fenced:
        reply = fenced.group(1)
    match = re.search(r"[\[{].*[\]}]", reply, re.S)
    if not match:
        raise AiError(
            f"The model's reply contained no JSON. Reply began: {reply.strip()[:120]!r}"
        )
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise AiError(f"The model's reply was not valid JSON: {exc}") from exc


CRITERIA_PROMPT = """You are helping set up a multi-criteria decision analysis.

Problem: {problem}

List the {n} most important criteria for this decision. Reply with JSON only,
as a list of objects with these keys:
  "name":      a short label
  "direction": "cost" if a smaller value is better, "benefit" if larger is
  "weight":    relative importance, a positive number
  "bounds":    [low, high] plausible range, or null if unknown

Reply with the JSON array and nothing else."""


def _parse_criteria(payload) -> tuple:
    """Turn a model's list into validated criteria, keeping the failures."""
    if not isinstance(payload, list):
        raise AiError(
            f"Expected a JSON list of criteria, got {type(payload).__name__}."
        )

    kept, rejected = [], []
    for entry in payload:
        name = (entry or {}).get("name") if isinstance(entry, dict) else None
        try:
            bounds = entry.get("bounds")
            if bounds is not None:
                bounds = tuple(bounds)
            kept.append(
                Criterion(
                    name=entry["name"],
                    weight=float(entry.get("weight", 1.0)),
                    direction=entry.get("direction", "benefit"),
                    bounds=bounds,
                )
            )
        except (McdaError, KeyError, TypeError, ValueError) as exc:
            rejected.append((name or "<unnamed>", str(exc)))

    if not kept:
        raise AiError(
            "No criterion in the model's reply was valid. Rejected: "
            + "; ".join(f"{n}: {w}" for n, w in rejected)
        )
    return tuple(kept), tuple(rejected)


def propose_criteria(
    problem: str,
    ask: Ask,
    n: int = 5,
    samples: int = 1,
) -> AiProposal:
    """Ask a model to suggest criteria for a described problem.

    Parameters
    ----------
    problem:
        A description in plain language.
    ask:
        Any callable mapping a prompt to the model's reply. The package
        imposes no vendor and acquires no dependency.
    samples:
        How many times to ask. Above one, the replies are compared and the
        share of criteria common to all is reported as ``agreement``. Asking
        twice costs twice as much and is usually worth it: a model that names
        different criteria each time is improvising, which no single reply
        would reveal.

    Returns
    -------
    AiProposal
        Validated criteria, the suggestions that failed validation and why,
        and the agreement across replies. Nothing is applied.
    """
    if samples < 1:
        raise McdaError("samples must be at least 1.")

    prompt = CRITERIA_PROMPT.format(problem=problem, n=n)
    replies = [ask(prompt) for _ in range(samples)]

    parsed = [_parse_criteria(_extract_json(reply)) for reply in replies]
    criteria, rejected = parsed[0]

    agreement = None
    if samples > 1:
        name_sets = [{c.name.strip().lower() for c in kept} for kept, _ in parsed]
        shared = set.intersection(*name_sets)
        union = set.union(*name_sets)
        agreement = len(shared) / len(union) if union else 0.0

    return AiProposal(
        criteria=criteria,
        rejected=rejected,
        agreement=agreement,
        raw=tuple(replies),
    )


WEIGHTS_PROMPT = """You are helping set up a multi-criteria decision analysis.

Problem: {problem}

Criteria, in order: {names}

Give each criterion a relative importance weight. Reply with JSON only: a
single object mapping each criterion name to a positive number.

Reply with the JSON object and nothing else."""


def propose_weights(
    problem: str,
    criteria: Sequence[Criterion],
    ask: Ask,
    matrix=None,
    samples: int = 1,
) -> WeightProposal:
    """Ask a model to weight known criteria, and check the answer.

    Where ``matrix`` is supplied, the proposed weights are used to rank the
    problem and the result is passed through
    :func:`~mcdakit.sensitivity`, so the caller learns not only what the model
    suggested but whether the resulting recommendation is fragile. The
    weights are additionally compared against the objective schemes of
    :mod:`mcdakit.weighting`; any that order the criteria differently are
    named in ``disagrees_with``.
    """
    if samples < 1:
        raise McdaError("samples must be at least 1.")
    criteria = tuple(criteria)
    names = ", ".join(c.name for c in criteria)

    prompt = WEIGHTS_PROMPT.format(problem=problem, names=names)
    replies = [ask(prompt) for _ in range(samples)]

    vectors = []
    for reply in replies:
        payload = _extract_json(reply)
        if not isinstance(payload, dict):
            raise AiError(
                f"Expected a JSON object of weights, got {type(payload).__name__}."
            )
        try:
            vector = np.array([float(payload[c.name]) for c in criteria], dtype=float)
        except KeyError as exc:
            raise AiError(
                f"The model omitted a weight for {exc.args[0]!r}. It replied "
                f"with: {sorted(payload)}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise AiError(f"A weight was not a number: {exc}") from exc

        if np.any(vector < 0) or not np.all(np.isfinite(vector)):
            raise AiError(
                "The model proposed a negative or non-finite weight; a "
                "criterion cannot count against itself."
            )
        if vector.sum() <= 0:
            raise AiError("The model proposed weights summing to zero.")
        vectors.append(vector / vector.sum())

    weights = vectors[0]

    agreement = None
    if samples > 1:
        # Mean pairwise cosine similarity: how consistently the model weights
        # the same problem. Reported rather than acted upon.
        sims = [
            float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            for i, a in enumerate(vectors)
            for b in vectors[i + 1 :]
        ]
        agreement = float(np.mean(sims)) if sims else None

    stability, disagrees = {}, ()
    if matrix is not None:
        stability, disagrees = _check_against_data(matrix, criteria, weights)

    return WeightProposal(
        weights=weights,
        criteria=criteria,
        stability=stability,
        disagrees_with=disagrees,
        agreement=agreement,
    )


def _check_against_data(matrix, criteria: Sequence[Criterion], weights):
    """Rank with the proposed weights, and compare them to objective schemes."""
    from .ranking import rank
    from .sensitivity import sensitivity
    from .types import Decision
    from .weighting import derive_weights

    weighted = [
        Criterion(c.name, float(w), c.direction, c.bounds)
        for c, w in zip(criteria, weights)
    ]
    decision = Decision(matrix, weighted)
    report = sensitivity(rank(decision))

    proposed_order = tuple(np.argsort(-weights))
    disagrees = []
    for scheme in ("entropy", "critic", "std"):
        try:
            objective = derive_weights(matrix, scheme, [c.direction for c in criteria])
        except McdaError:  # pragma: no cover - defensive
            continue
        if tuple(np.argsort(-objective)) != proposed_order:
            disagrees.append(scheme)

    return report, tuple(disagrees)


def critique_weights(
    weights: Sequence[float],
    criteria: Sequence[Criterion],
    matrix,
) -> dict:
    """Assess weights from any source against the data.

    Separated from :func:`propose_weights` so that weights elicited from a
    person, taken from a previous study, or produced by a model this package
    never saw can be held to the same standard. Returns the sensitivity report
    and the objective schemes that order the criteria differently.
    """
    vector = np.asarray(weights, dtype=float)
    criteria = tuple(criteria)
    if vector.shape != (len(criteria),):
        raise McdaError(f"Got {vector.shape} weights for {len(criteria)} criteria.")
    if np.any(vector < 0) or vector.sum() <= 0:
        raise McdaError("Weights must be non-negative and not all zero.")

    vector = vector / vector.sum()
    report, disagrees = _check_against_data(matrix, criteria, vector)
    return {
        "weights": vector,
        "stability": report,
        "disagrees_with": disagrees,
    }
