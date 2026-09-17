"""The extension contract: what a ranking method is.

A method is an object that turns a :class:`~mcdakit.types.Decision` into one
score per option, higher meaning better. Everything a method needs to say
about itself — its name, whether it wants the matrix oriented for it, whether
it can promise reversal-freedom — it says here, declaratively, so the library
never has to special-case it by name.

Writing one::

    from mcdakit import Method, ScoringContext, register
    import numpy as np

    class Median(Method):
        name = "median"
        summary = "Median score across criteria, ignoring weights."

        def score(self, ctx: ScoringContext) -> np.ndarray:
            return np.median(ctx.data, axis=1)

    register(Median())

See :doc:`/extending` for the full guide.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..types import Decision


class Wants(enum.Enum):
    """Which form of the decision matrix a method wants to be handed.

    Most methods assume larger is better and want cost criteria already
    flipped for them; a few — SPOTIS above all — must see the figures as
    measured, because they resolve direction themselves against fixed bounds.
    """

    #: Cost columns mirrored via ``max + min - x`` so higher is better
    #: throughout. The default, and what every classical method expects.
    ORIENTED = "oriented"

    #: The matrix exactly as the caller supplied it. The method is then
    #: responsible for reading ``ctx.directions`` itself.
    RAW = "raw"


@dataclass(frozen=True)
class ScoringContext:
    """Everything a method is given when asked to score.

    Bundling these into one object rather than a long parameter list is what
    lets the contract grow — a future field is additive, and no existing
    method's signature has to change to receive it.

    Attributes
    ----------
    data:
        The decision matrix, oriented or raw according to the method's
        :attr:`Method.wants`.
    weights:
        Criterion weights as the caller gave them, **unnormalised**. Methods
        that need them to sum to one should normalise, and must handle a total
        of zero; :attr:`normalized_weights` does both.
    decision:
        The original problem, for the rare method that needs something not
        surfaced here. Reach for this last: depending on it couples a method
        to the shape of :class:`~mcdakit.types.Decision`.
    options:
        Keyword arguments the caller passed through ``rank(..., **options)``.
    """

    data: np.ndarray
    weights: np.ndarray
    decision: Decision
    options: dict = field(default_factory=dict)

    #: The normalisation in force — the caller's override if given, else the
    #: method's declared default. Applied through :meth:`normalize`.
    _normalization: str | None = None

    # Names of the attributes a method has actually read. A mutable box on a
    # frozen dataclass, because observing a read must not require the context
    # to be rebuilt. Used to catch two mistakes that are otherwise silent:
    # options nobody consumed, and direction handled twice.
    _read: set = field(default_factory=set, repr=False, compare=False)

    @property
    def consumed(self) -> bool:
        """Whether anything has read :attr:`options` yet."""
        return "options" in self._read

    @property
    def read_directions(self) -> bool:
        """Whether the method inspected criterion directions itself."""
        return "directions" in self._read

    @property
    def opts(self) -> dict:
        """The caller's keyword arguments, marked as consumed.

        A method that accepts ``rank(..., v=0.5)``-style options must read them
        through this rather than :attr:`options` directly, so that an option
        nobody accepts is reported instead of vanishing.
        """
        self._read.add("options")
        return self.options

    @property
    def criteria(self) -> tuple:
        """The :class:`~mcdakit.types.Criterion` objects, in column order.

        These carry ``direction``, so a method reading them to decide
        cost-versus-benefit has the same double-handling risk as
        :attr:`directions`. Reading a criterion's ``bounds``,
        ``preference_shape`` or name is safe and common; only its
        ``direction`` is the trap.
        """
        return self.decision.criteria

    @property
    def directions(self) -> list:
        """``"benefit"`` or ``"cost"`` per criterion, in column order.

        Reading this while :attr:`Method.wants` is :attr:`Wants.ORIENTED` is
        almost always a bug: the cost columns have *already* been flipped for
        you, so handling direction again inverts them back. :func:`mcdakit.rank`
        raises if it sees that combination — see :class:`Wants`.
        """
        self._read.add("directions")
        return self.decision.directions

    @property
    def labels(self) -> tuple:
        return self.decision.labels

    @property
    def bounds(self) -> list:
        """``(lo, hi)`` or ``None`` per criterion, in column order."""
        return [c.bounds for c in self.decision.criteria]

    @property
    def n_options(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_criteria(self) -> int:
        return int(self.data.shape[1])

    def normalize(self, data: np.ndarray | None = None) -> np.ndarray:
        """Apply the normalisation in force for this call.

        The caller's ``normalization=`` if they gave one, otherwise the
        method's own default. Use this rather than normalising inline, so a
        method's scheme stays swappable and every zero-column guard lives in
        one place.
        """
        from ..normalization import normalize as _normalize

        matrix = self.data if data is None else data
        scheme = self.normalization
        if scheme is None:
            return np.asarray(matrix, dtype=float)
        return _normalize(matrix, scheme)

    @property
    def normalization(self):
        """The scheme in force: the caller's override, or the method's."""
        return self._normalization

    @property
    def normalized_weights(self) -> np.ndarray:
        """Weights scaled to sum to one, or uniform if the total is zero.

        A :class:`~mcdakit.types.Decision` refuses all-zero weights, so the
        fallback only fires for a method invoked on a hand-built context.
        """
        total = float(np.sum(self.weights))
        if total == 0:
            return np.full(self.n_criteria, 1.0 / self.n_criteria)
        return self.weights / total


@dataclass(frozen=True)
class MethodResult:
    """What a method returns when a bare score vector is not enough.

    Returning a plain ``np.ndarray`` from :meth:`Method.score` is equivalent to
    returning ``MethodResult(scores)``, and is what almost every method should
    do. Use this form to report a caveat the caller ought to see.

    Attributes
    ----------
    scores:
        One score per option, higher is better.
    reversal_free:
        ``True`` only if this ranking genuinely cannot change when an option is
        added or removed. Never set it optimistically: it is the strongest
        claim this library makes.
    warnings:
        Messages describing any way the result fell short of what the method
        normally promises.
    """

    scores: np.ndarray
    reversal_free: bool = False
    warnings: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "scores", np.asarray(self.scores, dtype=float))
        object.__setattr__(self, "warnings", tuple(self.warnings))


class Method:
    """Base class for a ranking method.

    Subclasses set :attr:`name` and implement :meth:`score`. Everything else
    has a working default.

    Attributes
    ----------
    name:
        The string callers pass as ``method=``. Must be unique across the
        registry; lowercase with underscores by convention.
    summary:
        One line describing the method, shown by
        :func:`~mcdakit.methods.registry.available`.
    citation:
        Where the method comes from. This is a research-adjacent library and
        will be read by people who know the papers — give them the reference.
    wants:
        Which form of the matrix :meth:`score` receives. See :class:`Wants`.
    reversal_free:
        Whether the method is rank-reversal-free *by construction*. Only
        SPOTIS sets this among the built-ins. A method whose guarantee is
        conditional should leave this ``False`` and return a
        :class:`MethodResult` saying so per call.
    """

    name: str = ""
    summary: str = ""
    citation: str = ""
    wants: Wants = Wants.ORIENTED
    reversal_free: bool = False

    #: Whether the method is defined only for strictly positive values.
    #: WASPAS divides by them and raises them to fractional powers; COPRAS
    #: normalises by a column total. Declaring the restriction lets the
    #: conformance suite check what the method actually promises rather than
    #: demanding a score for an input on which the method has no meaning.
    requires_positive: bool = False

    #: The normalisation this method is defined with — a name from
    #: :mod:`mcdakit.normalization`, or ``None`` for a method that does not
    #: normalise. Callers override it with ``rank(..., normalization=...)``,
    #: which is a real modelling choice: the ranking can change. Read it in
    #: :meth:`score` through ``ctx.normalize(data)``.
    normalization: str | None = None

    def score(self, ctx: ScoringContext):
        """Score every option. Higher must mean better.

        Return either an array of one score per option, or a
        :class:`MethodResult` when there is a caveat to report.

        A method whose natural measure is smallest-is-best — VIKOR's ``Q``,
        SPOTIS's distance — must negate before returning, and say so in its
        docstring. Getting this backwards produces a confident, exactly
        inverted ranking, which is the worst failure mode available here.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement score().")

    def validate(self, decision: Decision) -> None:
        """Raise if this method cannot handle the problem. Optional.

        Called before :meth:`score`. Use it to refuse a problem outright —
        a method needing at least three options, say. Prefer a
        :class:`MethodResult` warning for degradations that are merely
        regrettable rather than fatal.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name!r}>"


def _as_method_result(returned) -> MethodResult:
    """Normalise whatever :meth:`Method.score` returned."""
    if isinstance(returned, MethodResult):
        return returned
    return MethodResult(scores=returned)


__all__ = [
    "Method",
    "MethodResult",
    "ScoringContext",
    "Wants",
]
