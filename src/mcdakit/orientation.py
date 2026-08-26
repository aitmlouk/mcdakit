"""Turning cost criteria into benefit criteria.

Every classical method in this package is written for benefit criteria —
larger is better. Real decision matrices are mostly not: price and lead time
are the two most common criteria in procurement and both are costs. Without a
correction, a buyer entering genuine figures gets the most expensive, slowest
option ranked first: a confident and completely inverted answer.

Flipping once, here, rather than inside each algorithm keeps the method
implementations untouched and gives the correction a single place to be right.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def orient(data: np.ndarray, directions: Sequence[str]) -> np.ndarray:
    """Mirror every cost column so that higher is better throughout.

    The transform is ``max + min - x``, which reflects the column about its own
    midpoint. Two properties make it the right choice here:

    * it preserves the spacing between values, so methods that care about
      intervals see the same structure they would have seen on a benefit
      criterion;
    * it keeps everything non-negative when the input was, which TOPSIS's and
      ELECTRE's vector normalisation and SAW's division by the column maximum
      all rely on.

    Note it is *not* value-preserving across different option sets: the
    midpoint depends on which options are present. That dependence is one of
    the mechanisms behind rank reversal, and is precisely what SPOTIS avoids by
    working from fixed bounds instead — see
    :func:`mcdakit.methods.spotis.spotis`.
    """
    if data.size == 0:
        return data
    oriented = np.asarray(data, dtype=float).copy()
    for index, direction in enumerate(directions):
        if direction != "cost":
            continue
        column = oriented[:, index]
        oriented[:, index] = column.max() + column.min() - column
    return oriented
