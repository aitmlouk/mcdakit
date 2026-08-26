"""mcdakit — multi-criteria decision analysis whose rankings hold.

Most MCDA libraries answer "what is the ranking?". This one also answers
"would the ranking survive someone disagreeing with my inputs?", through two
features:

* :func:`~mcdakit.ranking.rank` with ``method="spotis"`` — a ranking that
  cannot change when an option nobody chose is removed from the shortlist.
* :func:`~mcdakit.sensitivity.sensitivity` — how far each weight can move
  before the winner changes, as a tolerance and a fragile/moderate/robust band.

Quick start
-----------

>>> from mcdakit import Criterion, rank, sensitivity
>>> criteria = [
...     Criterion("Price", weight=0.6, direction="cost", bounds=(2.0, 4.0)),
...     Criterion("Quality", weight=0.4, direction="benefit", bounds=(0, 10)),
... ]
>>> result = rank([[3.0, 8], [2.0, 6]], criteria,
...               method="spotis", labels=["Alpha", "Beta"])
>>> result.winner
'Beta'
>>> result.reversal_free
True

numpy is the only runtime dependency.
"""

from .ahp import (
    CONSISTENCY_LIMIT,
    RANDOM_INDEX,
    SAATY_SCALE,
    ahp_weights,
    comparison_matrix,
    consistency_ratio,
    priorities,
)
from .methods import (
    BUILTIN_METHODS,
    BoundsWarning,
    DuplicateMethodError,
    Method,
    MethodResult,
    PluginLoadError,
    ScoringContext,
    Wants,
    available,
    register,
    unregister,
)
from .methods import get as get_method
from .methods import has as has_method
from .methods import names as method_names
from .orientation import orient
from .ranking import (
    METHODS,
    agreement,
    compare_methods,
    rank,
    reversal_check,
    score,
)
from .sensitivity import (
    FRAGILE_THRESHOLD,
    ROBUST_THRESHOLD,
    sensitivity,
)
from .types import Criterion, Decision, McdaError, Result

# `testing` is imported lazily by users (`from mcdakit.testing import ...`);
# it is not pulled in here so `import mcdakit` stays free of test scaffolding.

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_METHODS",
    "CONSISTENCY_LIMIT",
    "FRAGILE_THRESHOLD",
    "METHODS",
    "RANDOM_INDEX",
    "ROBUST_THRESHOLD",
    "SAATY_SCALE",
    "BoundsWarning",
    "Criterion",
    "Decision",
    "DuplicateMethodError",
    "McdaError",
    "Method",
    "MethodResult",
    "PluginLoadError",
    "Result",
    "ScoringContext",
    "Wants",
    "__version__",
    "agreement",
    "ahp_weights",
    "available",
    "compare_methods",
    "comparison_matrix",
    "consistency_ratio",
    "get_method",
    "has_method",
    "method_names",
    "orient",
    "priorities",
    "rank",
    "register",
    "reversal_check",
    "score",
    "sensitivity",
    "unregister",
]
