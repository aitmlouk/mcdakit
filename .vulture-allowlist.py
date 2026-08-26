"""Names vulture cannot see are used, with the reason for each.

Run `vulture src/ .vulture-allowlist.py --min-confidence 60` to find dead code.
Anything listed here is reachable — vulture just cannot see the caller. Add an
entry only with a reason; an allowlist that accumulates unexplained names stops
being a check.
"""

from mcdakit.methods.registry import _reset_for_testing  # used by tests/
from mcdakit.types import Criterion, Result

Criterion.is_cost  # public API, exercised in tests/test_types.py
Result.score_of  # public API, used throughout the README and docs
_reset_for_testing  # test support, used in tests/test_registry.py
