"""The ranking methods, and the contract for adding more.

Each built-in exists in two forms: a plain function over numpy arrays, which
is what the algorithm actually is, and a :class:`~mcdakit.methods.base.Method`
class that registers it under a name. Use the functions directly when you want
the maths and nothing else; go through :func:`mcdakit.rank` for orientation,
weights, bounds and warning handling.

Adding your own is the same contract the built-ins use — see
:mod:`mcdakit.methods.base`.
"""

from .base import Method, MethodResult, ScoringContext, Wants
from .copras import Copras, copras
from .electre import Electre, electre
from .promethee import (
    Promethee,
    flows,
    preference_degree,
    preference_matrix,
    promethee,
    shapes_from,
)
from .reference_point import (
    Aras,
    Cocoso,
    Codas,
    Edas,
    Mabac,
    Marcos,
    aras,
    cocoso,
    codas,
    edas,
    mabac,
    marcos,
)
from .registry import (
    ENTRY_POINT_GROUP,
    DuplicateMethodError,
    PluginLoadError,
    available,
    get,
    has,
    names,
    register,
    unregister,
)
from .scoring import (
    Saw,
    SimpleScoring,
    WeightedScoring,
    saw,
    simple_scoring,
    weighted_scoring,
)
from .spotis import BoundsWarning, Spotis, spotis
from .topsis import Topsis, topsis
from .vikor import Vikor, vikor
from .waspas import Waspas, waspas

#: The built-ins, in a sensible reading order: simplest first, the
#: reversal-free one last.
BUILTIN_METHODS = (
    SimpleScoring,
    WeightedScoring,
    Saw,
    Topsis,
    Vikor,
    Electre,
    Promethee,
    Copras,
    Waspas,
    Aras,
    Cocoso,
    Codas,
    Edas,
    Mabac,
    Marcos,
    Spotis,
)


def register_builtins(*, replace: bool = False) -> None:
    """Register every built-in method.

    Called once on import. Exposed because
    :func:`~mcdakit.methods.registry._reset_for_testing` clears the registry,
    and tests need a way to put the built-ins back.
    """
    for method in BUILTIN_METHODS:
        register(method(), replace=replace)


register_builtins()

__all__ = [
    "BUILTIN_METHODS",
    "ENTRY_POINT_GROUP",
    "Aras",
    "BoundsWarning",
    "Cocoso",
    "Codas",
    "Copras",
    "DuplicateMethodError",
    "Edas",
    "Electre",
    "Mabac",
    "Marcos",
    "Method",
    "MethodResult",
    "PluginLoadError",
    "Promethee",
    "Saw",
    "ScoringContext",
    "SimpleScoring",
    "Spotis",
    "Topsis",
    "Vikor",
    "Wants",
    "Waspas",
    "WeightedScoring",
    "aras",
    "available",
    "cocoso",
    "codas",
    "copras",
    "edas",
    "electre",
    "flows",
    "get",
    "has",
    "mabac",
    "marcos",
    "names",
    "preference_degree",
    "preference_matrix",
    "promethee",
    "register",
    "register_builtins",
    "saw",
    "shapes_from",
    "simple_scoring",
    "spotis",
    "topsis",
    "unregister",
    "vikor",
    "waspas",
    "weighted_scoring",
]
