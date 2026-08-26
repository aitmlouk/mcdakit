"""Core input and output types.

Everything the caller builds by hand lives here. The types validate at
construction rather than at scoring time, so a mistake is reported where it
was made — a ragged matrix says it is ragged, instead of surfacing later as
"setting an array element with a sequence" from deep inside numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

DIRECTIONS = ("benefit", "cost")

PREFERENCE_SHAPES = ("usual", "ushape", "vshape", "level", "linear", "gaussian")


class McdaError(ValueError):
    """Raised when an input cannot describe a well-formed decision problem."""


@dataclass(frozen=True)
class Criterion:
    """One column of the decision matrix.

    Parameters
    ----------
    name:
        Human label, used in results and messages. Need not be unique, though
        duplicate names make a sensitivity report harder to read.
    weight:
        Relative importance. Any non-negative scale: weights are normalised to
        sum to one wherever a method needs them to, so 40/30/20/10 and
        0.4/0.3/0.2/0.1 are the same input.
    direction:
        ``"benefit"`` when larger values are better (quality, capacity),
        ``"cost"`` when smaller values are better (price, lead time).
    bounds:
        ``(lo, hi)`` for the range this criterion can take, independent of the
        options currently on the table — a budget ceiling, a rating scale, an
        acceptable delivery window. Optional for every method except SPOTIS,
        whose reversal-freedom depends on them; see
        :func:`mcdakit.methods.spotis.spotis`.
    preference_shape, q, p, s:
        PROMETHEE preference function and its thresholds. Ignored by the other
        methods. ``q`` is the indifference threshold, ``p`` the preference
        threshold, ``s`` the Gaussian parameter.
    """

    name: str
    weight: float = 1.0
    direction: str = "benefit"
    bounds: tuple | None = None
    preference_shape: str = "usual"
    q: float = 0.0
    p: float = 0.0
    s: float = 0.0

    def __post_init__(self):
        if not str(self.name).strip():
            raise McdaError("Every criterion needs a name.")

        weight = float(self.weight)
        if not np.isfinite(weight):
            raise McdaError(
                f"Criterion {self.name!r}: weight must be a finite number, "
                f"got {self.weight!r}."
            )
        if weight < 0:
            raise McdaError(
                f"Criterion {self.name!r}: weight must not be negative, "
                f"got {weight}."
            )
        object.__setattr__(self, "weight", weight)

        if self.direction not in DIRECTIONS:
            raise McdaError(
                f"Criterion {self.name!r}: direction must be one of "
                f"{', '.join(DIRECTIONS)}, got {self.direction!r}."
            )

        if self.bounds is not None:
            try:
                lo, hi = (float(b) for b in self.bounds)
            except (TypeError, ValueError) as exc:
                raise McdaError(
                    f"Criterion {self.name!r}: bounds must be a (lo, hi) pair, "
                    f"got {self.bounds!r}."
                ) from exc
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise McdaError(
                    f"Criterion {self.name!r}: bounds must be finite, "
                    f"got {self.bounds!r}."
                )
            if lo >= hi:
                raise McdaError(
                    f"Criterion {self.name!r}: bounds must have lo < hi, "
                    f"got ({lo}, {hi})."
                )
            object.__setattr__(self, "bounds", (lo, hi))

        if self.preference_shape not in PREFERENCE_SHAPES:
            raise McdaError(
                f"Criterion {self.name!r}: preference_shape must be one of "
                f"{', '.join(PREFERENCE_SHAPES)}, got {self.preference_shape!r}."
            )
        for label, value in (("q", self.q), ("p", self.p), ("s", self.s)):
            number = float(value)
            if number < 0 or not np.isfinite(number):
                raise McdaError(
                    f"Criterion {self.name!r}: {label} must be a non-negative "
                    f"finite number, got {value!r}."
                )
            object.__setattr__(self, label, number)
        if self.p and self.q and self.q >= self.p:
            raise McdaError(
                f"Criterion {self.name!r}: the preference threshold p must "
                f"exceed the indifference threshold q, got q={self.q}, "
                f"p={self.p}."
            )

    @property
    def is_cost(self) -> bool:
        return self.direction == "cost"


@dataclass(frozen=True)
class Decision:
    """A decision matrix, its criteria and its option labels.

    ``matrix`` is options (rows) by criteria (columns), holding the raw
    figures as measured — euros, days, ratings. Cost criteria are *not*
    pre-inverted by the caller; the methods handle direction themselves.
    """

    matrix: np.ndarray
    criteria: tuple
    labels: tuple

    def __init__(
        self,
        matrix,
        criteria: Sequence[Criterion],
        labels: Sequence[str] | None = None,
    ):
        criteria = tuple(criteria)
        if not criteria:
            raise McdaError("A decision needs at least one criterion.")
        for criterion in criteria:
            if not isinstance(criterion, Criterion):
                raise McdaError(
                    f"criteria must be Criterion instances, got "
                    f"{type(criterion).__name__}."
                )

        data = _as_matrix(matrix, len(criteria))

        if labels is None:
            labels = tuple(f"Option {i + 1}" for i in range(data.shape[0]))
        else:
            labels = tuple(str(label) for label in labels)
            if len(labels) != data.shape[0]:
                raise McdaError(
                    f"Got {len(labels)} labels for {data.shape[0]} options. "
                    f"There must be exactly one label per row of the matrix."
                )

        total = sum(c.weight for c in criteria)
        if total <= 0:
            raise McdaError(
                "Criterion weights are all zero, so no criterion can "
                "influence the result. Give at least one a positive weight."
            )

        object.__setattr__(self, "matrix", data)
        object.__setattr__(self, "criteria", criteria)
        object.__setattr__(self, "labels", labels)

    @property
    def weights(self) -> np.ndarray:
        """Weights as given, unnormalised."""
        return np.array([c.weight for c in self.criteria], dtype=float)

    @property
    def normalized_weights(self) -> np.ndarray:
        """Weights scaled to sum to one."""
        weights = self.weights
        return weights / weights.sum()

    @property
    def directions(self) -> list:
        return [c.direction for c in self.criteria]

    @property
    def names(self) -> list:
        return [c.name for c in self.criteria]

    @property
    def n_options(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n_criteria(self) -> int:
        return int(self.matrix.shape[1])

    def without(self, index: int) -> Decision:
        """The same problem with one option removed.

        The operation rank reversal is defined over: drop a loser, re-rank the
        survivors, and see whether their order held.
        """
        if not 0 <= index < self.n_options:
            raise McdaError(
                f"No option at index {index}; there are {self.n_options}."
            )
        keep = [i for i in range(self.n_options) if i != index]
        return Decision(
            self.matrix[keep, :],
            self.criteria,
            [self.labels[i] for i in keep],
        )


@dataclass(frozen=True)
class Result:
    """The outcome of ranking one decision under one method.

    ``scores`` is in the original option order and always reads
    higher-is-better, whatever sign convention the underlying method uses
    internally. ``ranking`` is sorted best first.
    """

    decision: Decision
    method: str
    scores: np.ndarray
    reversal_free: bool = False
    warnings: tuple = ()
    ranks: np.ndarray = field(default=None)

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=float)
        if scores.shape != (self.decision.n_options,):
            raise McdaError(
                f"Method {self.method!r} returned {scores.shape} scores for "
                f"{self.decision.n_options} options."
            )
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if self.ranks is None:
            object.__setattr__(self, "ranks", _competition_ranks(scores))

    @property
    def ranking(self) -> list:
        """``[(label, score), ...]`` best first."""
        order = sorted(
            range(len(self.scores)), key=lambda i: -self.scores[i]
        )
        return [(self.decision.labels[i], float(self.scores[i])) for i in order]

    @property
    def order(self) -> list:
        """Option labels best first."""
        return [label for label, _score in self.ranking]

    @property
    def winner(self) -> str:
        return self.order[0]

    @property
    def winner_index(self) -> int:
        return int(np.argmax(self.scores))

    def score_of(self, label: str) -> float:
        try:
            return float(self.scores[self.decision.labels.index(label)])
        except ValueError as exc:
            raise McdaError(f"No option named {label!r}.") from exc

    def rank_of(self, label: str) -> int:
        try:
            return int(self.ranks[self.decision.labels.index(label)])
        except ValueError as exc:
            raise McdaError(f"No option named {label!r}.") from exc

    def __str__(self) -> str:
        lines = [f"{self.method} ranking:"]
        width = max((len(label) for label in self.decision.labels), default=0)
        for label, score in self.ranking:
            rank = self.rank_of(label)
            lines.append(f"  {rank}. {label:<{width}}  {score: .4f}")
        for message in self.warnings:
            lines.append(f"  ! {message}")
        return "\n".join(lines)


def _as_matrix(matrix, n_criteria: int) -> np.ndarray:
    """Validate and convert a decision matrix to a float array."""
    if isinstance(matrix, np.ndarray):
        rows = matrix
    else:
        rows = list(matrix)
        lengths = {len(row) for row in rows} if rows else set()
        if len(lengths) > 1:
            raise McdaError(
                f"The decision matrix is ragged: rows have lengths "
                f"{sorted(lengths)}. Every option must be scored on every "
                f"criterion."
            )

    try:
        data = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as exc:
        raise McdaError(
            f"The decision matrix must contain numbers only: {exc}"
        ) from exc

    if data.ndim != 2:
        raise McdaError(
            f"The decision matrix must be two-dimensional (options x "
            f"criteria), got shape {data.shape}."
        )
    if data.shape[0] == 0:
        raise McdaError("A decision needs at least one option.")
    if data.shape[1] != n_criteria:
        raise McdaError(
            f"The decision matrix has {data.shape[1]} columns but "
            f"{n_criteria} criteria were given. There must be one column per "
            f"criterion, in the same order."
        )
    if not np.all(np.isfinite(data)):
        raise McdaError(
            "The decision matrix contains NaN or infinity. Every option must "
            "have a real score on every criterion."
        )
    return data


def _competition_ranks(scores: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Competition ranking (1-2-2-4), in the original option order.

    Two options tied for first are both 1st and the next is 3rd, never 2nd —
    the convention every sports table uses, and the one a reader assumes.
    """
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranks = np.zeros(len(scores), dtype=int)

    current_rank = 1
    tied_here = 0
    previous = None
    for position in order:
        score = scores[position]
        if previous is not None and abs(score - previous) < epsilon:
            ranks[position] = current_rank
        else:
            current_rank += tied_here
            tied_here = 0
            ranks[position] = current_rank
        previous = score
        tied_here += 1
    return ranks
