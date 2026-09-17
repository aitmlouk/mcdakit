"""One decision problem, put to every library, side by side.

The comparison worth making is not which library is fastest --- at these sizes
all of them are far below the point where speed affects an analysis --- but
what each one *tells you* about the same problem.

Run it with::

    python examples/same_problem_every_library.py

Requires the comparison libraries, which are development extras and never a
runtime dependency of mcdakit::

    pip install pymcdm pyDecision pyrepo-mcda
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

MATRIX = np.array(
    [
        [2.75, 7.0, 14.0, 8.0],
        [2.90, 8.5, 16.0, 8.0],
        [3.40, 9.0, 11.0, 7.0],
        [2.20, 3.0, 32.0, 2.0],
    ]
)
WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15])
DIRECTIONS = ["cost", "benefit", "cost", "benefit"]
TYPES = np.array([-1, 1, -1, 1])  # pymcdm / pyDecision convention
BOUNDS = np.array([[2.0, 4.0], [0.0, 10.0], [5.0, 35.0], [0.0, 10.0]])
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]


@contextlib.contextmanager
def quiet():
    """Some libraries print and plot on every call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def winner(scores, smaller_is_better=False):
    scores = np.asarray(scores, dtype=float)
    return LABELS[int(np.argmin(scores) if smaller_is_better else np.argmax(scores))]


def heading(text):
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# --------------------------------------------------------------------------
heading("Every library ranks it. They broadly agree.")
# --------------------------------------------------------------------------

from mcdakit import Criterion, Decision, rank  # noqa: E402

criteria = [
    Criterion(n, float(w), d, bounds=tuple(b))
    for n, w, d, b in zip(
        ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS, BOUNDS
    )
]
decision = Decision(MATRIX, criteria, LABELS)
ours = rank(decision, method="spotis")
print(f"  {'mcdakit (SPOTIS)':<28}{ours.winner}")

try:
    from pymcdm.methods import SPOTIS, TOPSIS

    print(
        f"  {'pymcdm (SPOTIS)':<28}"
        f"{winner(SPOTIS(BOUNDS)(MATRIX, WEIGHTS, TYPES), smaller_is_better=True)}"
    )
    print(f"  {'pymcdm (TOPSIS)':<28}{winner(TOPSIS()(MATRIX, WEIGHTS, TYPES))}")
except ImportError:
    print("  pymcdm not installed")

try:
    from pyrepo_mcda.mcda_methods import SPOTIS as RepoSPOTIS

    with quiet():
        # pyrepo-mcda expects bounds transposed, as 2 x n rather than n x 2.
        scores = RepoSPOTIS()(MATRIX, WEIGHTS, TYPES, BOUNDS.T)
    print(f"  {'pyrepo-mcda (SPOTIS)':<28}{winner(scores, smaller_is_better=True)}")
except Exception:
    print("  pyrepo-mcda not installed or incompatible signature")

print(
    "\n  So far nothing distinguishes them: four libraries, the same answer.\n"
    "  The difference is in what comes next."
)


# --------------------------------------------------------------------------
heading("Only one of them says whether that answer holds.")
# --------------------------------------------------------------------------

from mcdakit import sensitivity  # noqa: E402

report = sensitivity(ours)
print(
    f"  mcdakit      {ours.winner} wins, but the result is {report['level'].upper()}:"
)
print(
    f"               weighting {report['weakest']} "
    f"{report['overall']:.1%} differently would change it."
)
print()
print("  pymcdm       param_sensitivity() evaluates a grid the user supplies,")
print("               and returns the rankings obtained at each point.")
print("  pyrepo-mcda  likewise takes an array of percentage modifications.")
print("  pyDecision   no weight-sensitivity function.")
print()
print("  All three sample. None solves for the value at which the answer")
print("  changes, which is the number a decision report needs.")


# --------------------------------------------------------------------------
heading("Only one of them says why.")
# --------------------------------------------------------------------------

from mcdakit.explain import compare  # noqa: E402

print(compare(ours, ours.order[0], ours.order[1]))
print()
print("  No other library decomposes a margin by criterion.")


# --------------------------------------------------------------------------
heading("Only one of them refuses to answer when it cannot.")
# --------------------------------------------------------------------------

from mcdakit import McdaError  # noqa: E402

broken = MATRIX.copy()
broken[1, 2] = np.nan

try:
    rank(broken, criteria, labels=LABELS)
    print("  mcdakit      returned a ranking")
except McdaError as exc:
    print(f"  mcdakit      refused: {str(exc)[:58]}...")

try:
    from pymcdm.methods import TOPSIS

    with quiet():
        scores = TOPSIS()(broken, WEIGHTS, TYPES)
    state = "NaN" if not np.all(np.isfinite(scores)) else "a value"
    print(f"  pymcdm       returned {state}")
except ImportError:
    pass

print(
    "\n  A ranking containing NaN is not a ranking: NaN compares false against\n"
    "  everything, so the order depends on the sorting algorithm."
)
