"""Run one decision problem through every library that implements each method.

The purpose is correctness, not comparison. Agreeing with another library does
not by itself prove either is right, so the evidence is ranked:

1. a published worked example, where the method's authors give the expected
   numbers;
2. a hand computation, verifiable on paper;
3. agreement between independent implementations, which is weak alone but
   meaningful when several converge.

Where two libraries disagree the difference is either a defect in one of them
or a documented modelling choice, and this script's job is to surface it so
that somebody decides which. A silent disagreement is the failure mode worth
avoiding: TOPSIS appeared to disagree with ``pymcdm`` until the cause was
identified as min--max versus vector normalisation.

Run with::

    python benchmarks/crosscheck.py

Requires the comparison libraries, which are dev-only and never a runtime
dependency::

    pip install pymcdm pyDecision
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

# --------------------------------------------------------------------------
# One shared problem, used by every method below.
#
# Four suppliers on two cost criteria (price, lead time) and two benefit
# criteria (quality, support). Deliberately not symmetric: a matrix where the
# same alternative wins under every method would hide a disagreement rather
# than reveal one.
# --------------------------------------------------------------------------
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
#: pymcdm and pyDecision both encode direction as +1 profit / -1 cost.
TYPES = np.array([-1, 1, -1, 1])
BOUNDS = np.array([[2.0, 4.0], [0.0, 10.0], [5.0, 35.0], [0.0, 10.0]])
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]


def order(scores, smaller_is_better=False):
    """Positions best-first, so libraries with opposite sign conventions can
    still be compared on the thing that matters."""
    scores = np.asarray(scores, dtype=float)
    return tuple(np.argsort(scores if smaller_is_better else -scores))


def order_indexed(table, smaller_is_better=False):
    """Best-first order from a pyDecision result table.

    pyDecision returns ``[[option_number, value], ...]`` **already sorted by
    value**, so reading column 1 positionally attributes each score to the
    wrong alternative. Column 0 carries the 1-based option number and is the
    only safe way to recover the mapping. Missing this produced an apparent
    disagreement on VIKOR that was entirely an artefact of this harness.
    """
    table = np.asarray(table, dtype=float)
    values = table[:, 1]
    indices = table[:, 0].astype(int) - 1
    ranked = indices[np.argsort(values if smaller_is_better else -values)]
    return tuple(ranked)


def named(positions):
    return [LABELS[i] for i in positions]


@contextlib.contextmanager
def quiet():
    """Silence rival libraries that print and plot on every call."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            yield


def ours(method, **kwargs):
    from mcdakit import Criterion, Decision, rank

    criteria = [
        Criterion(name, float(w), d, bounds=tuple(b))
        for name, w, d, b in zip(
            ("Price", "Quality", "Lead time", "Support"),
            WEIGHTS,
            DIRECTIONS,
            BOUNDS,
        )
    ]
    return rank(Decision(MATRIX, criteria, LABELS), method=method, **kwargs).scores


# --------------------------------------------------------------------------
# One entry per (method, rival). Each returns a best-first ordering.
# --------------------------------------------------------------------------


def _pymcdm_topsis_vector():
    from pymcdm.methods import TOPSIS
    from pymcdm.normalizations import vector_normalization

    return order(
        TOPSIS(normalization_function=vector_normalization)(MATRIX, WEIGHTS, TYPES)
    )


def _pymcdm_topsis_default():
    from pymcdm.methods import TOPSIS

    return order(TOPSIS()(MATRIX, WEIGHTS, TYPES))


def _pymcdm_spotis():
    from pymcdm.methods import SPOTIS

    return order(SPOTIS(BOUNDS)(MATRIX, WEIGHTS, TYPES), smaller_is_better=True)


def _pymcdm_vikor():
    from pymcdm.methods import VIKOR

    return order(VIKOR()(MATRIX, WEIGHTS, TYPES), smaller_is_better=True)


def _pymcdm_promethee():
    from pymcdm.methods import PROMETHEE_II

    return order(PROMETHEE_II("usual")(MATRIX, WEIGHTS, TYPES))


def _pydecision_saw():
    from pyDecision.algorithm import saw_method

    with quiet():
        scores = saw_method(
            MATRIX,
            ["min" if d == "cost" else "max" for d in DIRECTIONS],
            WEIGHTS,
            graph=False,
            verbose=False,
        )
    return order_indexed(scores)


def _pydecision_vikor():
    from pyDecision.algorithm import vikor_method

    with quiet():
        out = vikor_method(
            MATRIX,
            WEIGHTS,
            ["min" if d == "cost" else "max" for d in DIRECTIONS],
            graph=False,
            verbose=False,
        )
    # vikor_method returns (S, R, Q, order); Q is the compromise measure,
    # already sorted, with the option number in column 0.
    return order_indexed(out[2], smaller_is_better=True)


#: ``method -> {rival label: callable}``. A rival absent for a method simply
#: has no entry; the report says so rather than inventing a comparison.
CHECKS = {
    "topsis": {
        "pymcdm (vector norm)": _pymcdm_topsis_vector,
        "pymcdm (default min-max)": _pymcdm_topsis_default,
    },
    "spotis": {"pymcdm": _pymcdm_spotis},
    "vikor": {"pymcdm": _pymcdm_vikor, "pyDecision": _pydecision_vikor},
    "promethee": {"pymcdm (usual)": _pymcdm_promethee},
    "saw": {"pyDecision": _pydecision_saw},
}

#: Differences already investigated and attributed, so that a known modelling
#: choice is not re-reported as a fresh discrepancy each run.
EXPECTED_DIFFERENCES = {
    ("saw", "pyDecision"): (
        "the two implement different standard SAW variants for cost criteria: "
        "pyDecision inverts the ratio (min/x) where mcdakit mirrors the "
        "interval (max+min-x) before dividing by the column maximum. An "
        "alternative with no cost advantage scores identically under both "
        "(Option 3: 0.840074), which isolates the difference to the cost "
        "transform."
    ),
    ("topsis", "pymcdm (default min-max)"): (
        "pymcdm defaults to min-max normalisation; mcdakit follows Hwang and "
        "Yoon's vector normalisation. Matching the normalisation makes them "
        "agree, as the row above shows."
    ),
}


def main() -> int:
    print("=" * 74)
    print("Cross-checking mcdakit against independent implementations")
    print("=" * 74)

    import mcdakit

    print(f"\nmcdakit {mcdakit.__version__}")
    for name in ("pymcdm", "pyDecision"):
        try:
            module = __import__(name)
            print(f"{name} {getattr(module, '__version__', 'installed')}")
        except Exception:  # pragma: no cover
            print(f"{name} not installed")

    unexplained = []
    for method, rivals in CHECKS.items():
        mine = order(ours(method), smaller_is_better=False)
        print(f"\n{'-' * 74}\n{method}\n{'-' * 74}")
        print(f"  mcdakit              {named(mine)}")
        for label, fn in rivals.items():
            try:
                theirs = fn()
            except Exception as exc:
                print(f"  {label:<21}unavailable: {type(exc).__name__}: {exc}")
                continue
            verdict = "MATCH " if theirs == mine else "DIFFER"
            print(f"  {label:<21}{named(theirs)}  {verdict}")
            if theirs != mine:
                note = EXPECTED_DIFFERENCES.get((method, label))
                if note:
                    print(f"    explained: {note}")
                else:
                    unexplained.append((method, label))

    print(f"\n{'=' * 74}")
    if unexplained:
        print("UNEXPLAINED DIFFERENCES — each is a defect until attributed:")
        for method, label in unexplained:
            print(f"  {method} vs {label}")
        return 1
    print("Every difference is accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
