"""The ranking methods.

Each is a plain function over numpy arrays returning a score per option where
higher is better. They can be used directly, but the usual entry point is
:func:`mcdakit.rank`, which handles orientation, weights and bounds.
"""

from .electre import electre
from .promethee import flows, preference_degree, preference_matrix, promethee
from .scoring import saw, simple_scoring, weighted_scoring
from .spotis import BoundsWarning, spotis
from .topsis import topsis
from .vikor import vikor

__all__ = [
    "BoundsWarning",
    "electre",
    "flows",
    "preference_degree",
    "preference_matrix",
    "promethee",
    "saw",
    "simple_scoring",
    "spotis",
    "topsis",
    "vikor",
    "weighted_scoring",
]
