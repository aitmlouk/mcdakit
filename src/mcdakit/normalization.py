"""Normalisation schemes, and how to add your own.

Normalisation is not a preprocessing detail. Putting a price spanning 900 and
a rating spanning 3 onto a common scale is a *modelling* decision: which
scheme you pick changes the numbers, sometimes the ranking, and — as the
rank-reversal literature shows — how stable that ranking is when the option
set changes. Two methods that differ only in normalisation are two different
methods.

So the schemes are named, documented, cited and swappable, rather than
hardcoded inside each algorithm:

    >>> from mcdakit import rank, Criterion
    >>> criteria = [Criterion("A", 1.0), Criterion("B", 1.0)]
    >>> a = rank([[1.0, 9.0], [9.0, 1.0]], criteria, method="topsis")
    >>> b = rank([[1.0, 9.0], [9.0, 1.0]], criteria, method="topsis",
    ...          normalization="minmax")
    >>> a.method, b.method
    ('topsis', 'topsis')

Every scheme takes a matrix that is already oriented — larger is better — and
returns one of the same shape. A column carrying no information (every option
scoring alike, or all zeros) is given a constant rather than a division by
zero: it cannot discriminate, and saying so with a flat value is honest where
a NaN would poison the whole ranking.

Adding one is the same shape as adding a method::

    from mcdakit import register_normalization

    @register_normalization("softmax", summary="Exponential, sums to one.")
    def softmax(data):
        shifted = np.exp(data - data.max(axis=0))
        return shifted / shifted.sum(axis=0)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .types import McdaError

__all__ = [
    "NORMALIZATIONS",
    "available_normalizations",
    "get_normalization",
    "max_normalization",
    "minmax_normalization",
    "normalize_weights",
    "register_normalization",
    "sum_normalization",
    "vector_normalization",
]


def _safe(divisor: np.ndarray) -> np.ndarray:
    """Replace a zero divisor with one.

    A column whose divisor is zero carries no information — every option
    scores the same, or all score nothing. Dividing would give NaN, which
    sorts unpredictably and would silently corrupt the ranking of every other
    criterion too.
    """
    return np.where(divisor == 0, 1.0, divisor)


def vector_normalization(data: np.ndarray) -> np.ndarray:
    """``x / sqrt(sum(x^2))`` — Hwang and Yoon's original choice for TOPSIS.

    Preserves the ratios between values and is insensitive to the units the
    column was measured in. Because the denominator depends on every value in
    the column, adding or removing an option changes every other option's
    normalised score.

    Hwang and Yoon, *Multiple Attribute Decision Making*, 1981.
    """
    return data / _safe(np.sqrt(np.sum(data**2, axis=0)))


def minmax_normalization(data: np.ndarray) -> np.ndarray:
    """``(x - min) / (max - min)`` — the observed range mapped onto ``[0, 1]``.

    The most intuitive scheme and the most reversal-prone: the best option
    always scores 1 and the worst always 0, so removing either rescales
    everyone else. This is the mechanism behind most of the rank reversal this
    package measures.
    """
    col_min = np.min(data, axis=0)
    return (data - col_min) / _safe(np.max(data, axis=0) - col_min)


def max_normalization(data: np.ndarray) -> np.ndarray:
    """``x / max`` — SAW's scheme.

    Keeps ratios intact and does not pin the worst option to zero. More stable
    than min-max under option removal, because only the maximum matters and
    the option removed is usually not it.
    """
    return data / _safe(np.max(data, axis=0))


def sum_normalization(data: np.ndarray) -> np.ndarray:
    """``x / sum(x)`` — each column becomes a share of its total.

    Used by COPRAS and several outranking methods. Every option influences
    every other's score, so it is sensitive to the option set.
    """
    return data / _safe(np.sum(data, axis=0))


#: Name to ``(function, summary)``. Insertion-ordered; plugins append.
NORMALIZATIONS: dict = {}


def register_normalization(
    name: str,
    function: Callable | None = None,
    *,
    summary: str = "",
    replace: bool = False,
):
    """Register a normalisation scheme. Usable as a decorator.

    Parameters
    ----------
    name:
        What callers pass as ``normalization=``.
    function:
        Takes an oriented matrix, returns one of the same shape. Omit to use
        as a decorator.
    replace:
        Overwrite an existing scheme of this name. Off by default, for the
        same reason methods refuse it: silently shadowing one would make the
        same call mean different things in different processes.
    """

    def _register(func: Callable) -> Callable:
        if not callable(func):
            raise McdaError(
                f"A normalisation must be callable, got {type(func).__name__}."
            )
        if not name or not name.strip():
            raise McdaError("A normalisation needs a non-empty name.")
        if name in NORMALIZATIONS and not replace:
            raise McdaError(
                f"A normalisation named {name!r} is already registered. "
                f"Choose another name, or pass replace=True to override it "
                f"deliberately."
            )
        NORMALIZATIONS[name] = (
            func,
            summary or (func.__doc__ or "").strip().split("\n")[0],
        )
        return func

    return _register if function is None else _register(function)


def get_normalization(name: str) -> Callable:
    """Look up a scheme by name, or raise listing what is available."""
    try:
        return NORMALIZATIONS[name][0]
    except KeyError:
        raise McdaError(
            f"Unknown normalization {name!r}. Available: "
            f"{', '.join(sorted(NORMALIZATIONS))}."
        ) from None


def available_normalizations() -> dict:
    """``{name: summary}`` for every registered scheme."""
    return {name: summary for name, (_f, summary) in NORMALIZATIONS.items()}


def normalize(data: np.ndarray, scheme) -> np.ndarray:
    """Apply a scheme given by name or as a callable.

    Accepting a bare callable means a one-off scheme needs no registration —
    registration is for schemes you want to name and share.
    """
    function = get_normalization(scheme) if isinstance(scheme, str) else scheme
    if not callable(function):
        raise McdaError(
            f"normalization must be a name or a callable, got {type(scheme).__name__}."
        )
    result = np.asarray(function(np.asarray(data, dtype=float)), dtype=float)
    if result.shape != np.shape(data):
        raise McdaError(
            f"Normalisation {getattr(function, '__name__', scheme)!r} changed "
            f"the matrix shape from {np.shape(data)} to {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise McdaError(
            f"Normalisation {getattr(function, '__name__', scheme)!r} produced "
            f"a non-finite value. A column that cannot discriminate should be "
            f"given a constant, not divided by zero."
        )
    return result


register_normalization(
    "vector",
    vector_normalization,
    summary="x / sqrt(sum(x^2)); unit-insensitive, preserves ratios.",
)
register_normalization(
    "minmax",
    minmax_normalization,
    summary="(x - min) / (max - min); observed range onto [0, 1].",
)
register_normalization(
    "max",
    max_normalization,
    summary="x / max; preserves ratios, worst option not pinned to zero.",
)
register_normalization(
    "sum",
    sum_normalization,
    summary="x / sum(x); each value as a share of the column total.",
)


def normalize_weights(weights: np.ndarray) -> np.ndarray | None:
    """Scale weights to sum to one, or return ``None`` if the total is zero.

    Every method needs this, and each carried its own copy of the guard.
    ``None`` rather than a fallback keeps the decision with the caller: the
    methods return a flat zero vector, which says "nothing here can be ranked"
    rather than inventing preferences nobody expressed.

    Going through :func:`mcdakit.rank` the zero case cannot arise — a
    :class:`~mcdakit.types.Decision` refuses all-zero weights. The guard is for
    the algorithm functions, which are public and callable on hand-built
    arrays.
    """
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    return None if total == 0 else weights / total
