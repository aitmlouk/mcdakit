"""Deriving criterion weights from the data.

Weights are the softest input in any MCDA model. :func:`~mcdakit.ahp_weights`
elicits them from a person, which is the right approach when someone genuinely
holds a view about what matters. Often nobody does, or the analyst wants a
starting point that is not an opinion — and the field's answer is to read the
weights off the decision matrix itself.

The premise of every scheme here: **a criterion on which all the options score
alike cannot separate them, so it carries no information and deserves little
weight.** They differ in how they measure that, and in whether they account
for criteria that duplicate each other.

    >>> import numpy as np
    >>> from mcdakit import entropy_weights
    >>> matrix = np.array([[5.0, 1.0], [5.0, 9.0], [5.0, 5.0]])
    >>> w = entropy_weights(matrix)
    >>> bool(w[0] < 1e-9), round(float(w[1]), 4)
    (True, 1.0)

The first criterion is identical for every option, so it gets no weight at
all. That is the point.

These are **objective** in the sense of being computed rather than asserted,
not in the sense of being right. A criterion can be uninformative in the
current shortlist and still be the one that matters — a budget everyone
happens to meet is not thereby unimportant. Treat these as a defensible
default and a check on elicited weights, never as a replacement for knowing
what the decision is about.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import numpy as np

from .orientation import orient
from .types import McdaError

__all__ = [
    "WEIGHTINGS",
    "available_weightings",
    "critic_weights",
    "derive_weights",
    "entropy_weights",
    "equal_weights",
    "get_weighting",
    "register_weighting",
    "std_weights",
]


def _prepare(matrix, directions: Sequence[str] | None = None) -> np.ndarray:
    """Validate the matrix and orient it so larger is better."""
    data = np.asarray(matrix, dtype=float)
    if data.ndim != 2:
        raise McdaError(
            f"A decision matrix must be two-dimensional (options x criteria), "
            f"got shape {data.shape}."
        )
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise McdaError(
            "A decision matrix needs at least one option and one criterion."
        )
    if not np.all(np.isfinite(data)):
        raise McdaError(
            "The decision matrix contains NaN or infinity; weights cannot be "
            "derived from it."
        )
    if directions is not None:
        if len(directions) != data.shape[1]:
            raise McdaError(
                f"Got {len(directions)} directions for {data.shape[1]} criteria."
            )
        data = orient(data, directions)
    return data


def _normalise(weights: np.ndarray) -> np.ndarray:
    """Scale to sum to one, falling back to equal weights.

    The fallback fires when every criterion scored zero information — a matrix
    where no criterion separates anything. Equal weights are the only honest
    reading of that, and beat dividing by zero.
    """
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def equal_weights(matrix, directions: Sequence[str] | None = None) -> np.ndarray:
    """Every criterion equally important.

    The honest default when nobody has expressed a preference, and the
    baseline the others should be compared against: if an objective scheme
    does not beat equal weights on your problem, it is adding complexity
    rather than information.
    """
    data = _prepare(matrix, directions)
    return np.full(data.shape[1], 1.0 / data.shape[1])


def entropy_weights(matrix, directions: Sequence[str] | None = None) -> np.ndarray:
    """Shannon entropy weights.

    Normalise each column to a probability distribution, measure its entropy,
    and weight by how far that falls short of maximum. A column where every
    option scores alike has maximum entropy and therefore zero weight; a column
    that separates the options sharply has low entropy and high weight.

    .. math::

        p_{ij} = \\frac{x_{ij}}{\\sum_i x_{ij}}
        \\qquad
        e_j = -k \\sum_i p_{ij} \\ln p_{ij}
        \\qquad
        w_j \\propto 1 - e_j

    with :math:`k = 1/\\ln m` so that :math:`e_j \\in [0, 1]`.

    Values are shifted to be non-negative first, since the logarithm requires
    it and a criterion measured on a scale that goes negative — a temperature,
    a profit-or-loss — is otherwise unusable.

    Shannon (1948); the MCDA application is due to Zeleny (1982).
    """
    data = _prepare(matrix, directions)
    n_options = data.shape[0]

    if n_options < 2:
        # With one option there is nothing to separate, so no column can be
        # more informative than any other.
        return equal_weights(data)

    # Shift to non-negative rather than subtracting the column minimum: a
    # constant column would otherwise become all zeros, which reads as zero
    # entropy and so *maximum* weight — the exact inverse of what the method
    # means. Shifting by the global minimum keeps a flat column flat and
    # positive, where it correctly reaches maximum entropy.
    floor = float(data.min())
    shifted = data - floor + 1.0 if floor <= 0 else data.copy()

    totals = shifted.sum(axis=0)
    p = shifted / np.where(totals == 0, 1.0, totals)

    # 0 log 0 is 0 in the limit; numpy would return nan.
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)

    k = 1.0 / np.log(n_options)
    entropy = -k * terms.sum(axis=0)

    # Floating point can leave entropy a hair above 1 for a perfectly flat
    # column, which would make its weight negative.
    return _normalise(np.clip(1.0 - entropy, 0.0, None))


def std_weights(matrix, directions: Sequence[str] | None = None) -> np.ndarray:
    """Standard-deviation weights.

    Weight each criterion by how much the options vary on it. The simplest
    objective scheme, and the easiest to explain: a criterion everyone scores
    the same tells you nothing, so it gets nothing.

    Unlike :func:`entropy_weights` it is sensitive to the units a criterion is
    measured in — a price in euros varies more than the same price in
    thousands — so it suits matrices already on comparable scales, such as
    ratings, better than raw mixed units.
    """
    data = _prepare(matrix, directions)
    return _normalise(data.std(axis=0))


def critic_weights(matrix, directions: Sequence[str] | None = None) -> np.ndarray:
    """CRITIC — CRiteria Importance Through Intercriteria Correlation.

    Weights by contrast *and* independence: a criterion earns weight for
    varying a lot, and loses it for saying the same thing as the others.

    .. math::

        C_j = \\sigma_j \\sum_k (1 - r_{jk})

    where :math:`\\sigma_j` is the standard deviation of the min-max normalised
    column and :math:`r_{jk}` the correlation between columns *j* and *k*.

    This is what distinguishes it from :func:`std_weights`: two criteria
    measuring nearly the same thing — list price and total cost, say — would
    each get full weight under a variance-only scheme, effectively counting
    that dimension twice. CRITIC splits it between them.

    Diakoulaki, Mavrotas and Papayannakis, *Determining objective weights in
    multiple criteria problems: the CRITIC method*, Computers & Operations
    Research 22(7), 1995.
    """
    data = _prepare(matrix, directions)
    n_options, n_criteria = data.shape

    if n_options < 2:
        return equal_weights(data)

    span = data.max(axis=0) - data.min(axis=0)
    span = np.where(span == 0, 1.0, span)
    normalised = (data - data.min(axis=0)) / span

    sigma = normalised.std(axis=0)

    if n_criteria == 1:
        return np.ones(1)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(normalised, rowvar=False)
    # A constant column has zero variance, so its correlation is undefined.
    # Treating it as uncorrelated is the neutral choice: it neither duplicates
    # another criterion nor is duplicated by one.
    correlation = np.nan_to_num(correlation, nan=0.0)

    conflict = (1.0 - correlation).sum(axis=1)
    return _normalise(sigma * conflict)


#: Name to ``(function, summary)``. Insertion-ordered; plugins append.
WEIGHTINGS: dict = {}


def register_weighting(
    name: str,
    function: Callable | None = None,
    *,
    summary: str = "",
    replace: bool = False,
):
    """Register a weighting scheme. Usable as a decorator.

    A scheme takes ``(matrix, directions)`` and returns one weight per
    criterion. It need not normalise — :func:`derive_weights` does that — but
    it must return non-negative, finite values.
    """

    def _register(func: Callable) -> Callable:
        if not callable(func):
            raise McdaError(f"A weighting must be callable, got {type(func).__name__}.")
        if not name or not name.strip():
            raise McdaError("A weighting needs a non-empty name.")
        if name in WEIGHTINGS and not replace:
            raise McdaError(
                f"A weighting named {name!r} is already registered. Choose "
                f"another name, or pass replace=True to override it "
                f"deliberately."
            )
        WEIGHTINGS[name] = (
            func,
            summary or (func.__doc__ or "").strip().split("\n")[0],
        )
        return func

    return _register if function is None else _register(function)


def get_weighting(name: str) -> Callable:
    """Look up a scheme by name, or raise listing what is available."""
    try:
        return WEIGHTINGS[name][0]
    except KeyError:
        raise McdaError(
            f"Unknown weighting {name!r}. Available: {', '.join(sorted(WEIGHTINGS))}."
        ) from None


def available_weightings() -> dict:
    """``{name: summary}`` for every registered scheme."""
    return {name: summary for name, (_f, summary) in WEIGHTINGS.items()}


def derive_weights(
    matrix,
    scheme: str = "entropy",
    directions: Sequence[str] | None = None,
) -> np.ndarray:
    """Weights from a decision matrix, by name or by callable.

    Returns one weight per criterion, summing to one.

    Parameters
    ----------
    directions:
        Optional ``"benefit"``/``"cost"`` per criterion. Supplying them
        orients the matrix first, which matters for :func:`critic_weights`,
        where the sign of a correlation depends on it: a cost and a benefit
        criterion that move together look opposed until one is flipped.

    Examples
    --------
    >>> from mcdakit import derive_weights
    >>> matrix = [[2.0, 8.0], [3.0, 6.0], [4.0, 4.0]]
    >>> w = derive_weights(matrix, "equal")
    >>> [round(float(x), 3) for x in w]
    [0.5, 0.5]
    """
    function = get_weighting(scheme) if isinstance(scheme, str) else scheme
    if not callable(function):
        raise McdaError(
            f"scheme must be a name or a callable, got {type(scheme).__name__}."
        )

    weights = np.asarray(function(matrix, directions), dtype=float)
    n_criteria = np.asarray(matrix, dtype=float).shape[1]

    if weights.shape != (n_criteria,):
        raise McdaError(
            f"Weighting {getattr(function, '__name__', scheme)!r} returned "
            f"{weights.shape} weights for {n_criteria} criteria."
        )
    if not np.all(np.isfinite(weights)):
        raise McdaError(
            f"Weighting {getattr(function, '__name__', scheme)!r} produced a "
            f"non-finite weight."
        )
    if np.any(weights < 0):
        raise McdaError(
            f"Weighting {getattr(function, '__name__', scheme)!r} produced a "
            f"negative weight. A criterion cannot count against itself."
        )
    return _normalise(weights)


register_weighting("equal", equal_weights, summary="Every criterion equally important.")
register_weighting(
    "entropy",
    entropy_weights,
    summary="Shannon entropy; weight by how much a criterion separates the options.",
)
register_weighting("std", std_weights, summary="Standard deviation; weight by spread.")
register_weighting(
    "critic",
    critic_weights,
    summary="CRITIC; weight by contrast and independence from other criteria.",
)
