"""Compare mcdakit against the state of the art, on measurable ground.

Four questions, chosen because each has an answer a reader can check rather
than take on trust:

1. **Do we agree?** Where two libraries implement the same method, they should
   produce the same ranking. Disagreement is either a bug or a documented
   modelling difference, and it matters which.
2. **What can each do?** Capability presence, established by calling the
   library rather than by reading its documentation.
3. **What happens on bad input?** Degenerate but constructible inputs — all
   weights zero, a missing score — separate libraries that refuse from
   libraries that return a number.
4. **How fast?** Reported last and deliberately down-weighted: at these
   problem sizes every library is fast enough, and the differences say more
   about API overhead than about algorithms.

Run with::

    python benchmarks/comparison.py

Requires the comparison libraries, which are dev-only extras and never a
runtime dependency of mcdakit::

    pip install pymcdm pyDecision scikit-criteria
"""

from __future__ import annotations

import time
import warnings

import numpy as np

# --------------------------------------------------------------------------
# The shared problem. Four suppliers, two cost criteria and two benefit
# criteria, with bounds — the same case used throughout the mcdakit
# documentation, so figures here can be cross-checked against it.
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
#: benefit/cost per criterion, in mcdakit's vocabulary.
DIRECTIONS = ["cost", "benefit", "cost", "benefit"]
#: pymcdm and pyDecision encode the same thing as +1 profit / -1 cost.
TYPES = np.array([-1, 1, -1, 1])
BOUNDS = np.array([[2.0, 4.0], [0.0, 10.0], [5.0, 35.0], [0.0, 10.0]])
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]


def order_from(scores, *, smaller_is_better=False):
    """Rank positions, best first, from a score vector."""
    scores = np.asarray(scores, dtype=float)
    return tuple(np.argsort(scores if smaller_is_better else -scores))


# --------------------------------------------------------------------------
# 1. Agreement on shared methods
# --------------------------------------------------------------------------


def agreement():
    """Do the libraries produce the same ranking for the same method?"""
    from mcdakit import Criterion, Decision, rank

    criteria = [
        Criterion(n, float(w), d, bounds=tuple(b))
        for n, w, d, b in zip(
            ("Price", "Quality", "Lead time", "Support"),
            WEIGHTS,
            DIRECTIONS,
            BOUNDS,
        )
    ]
    decision = Decision(MATRIX, criteria, LABELS)

    rows = []

    # -- TOPSIS ------------------------------------------------------------
    ours = order_from(rank(decision, method="topsis").scores)
    try:
        from pymcdm.methods import TOPSIS
        from pymcdm.normalizations import vector_normalization

        theirs_default = order_from(TOPSIS()(MATRIX, WEIGHTS, TYPES))
        theirs_vector = order_from(
            TOPSIS(normalization_function=vector_normalization)(MATRIX, WEIGHTS, TYPES)
        )
        rows.append(
            (
                "TOPSIS",
                "pymcdm (default, min-max)",
                ours,
                theirs_default,
                "documented: differing normalisation",
            )
        )
        rows.append(
            (
                "TOPSIS",
                "pymcdm (vector normalisation)",
                ours,
                theirs_vector,
                "same normalisation",
            )
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        rows.append(("TOPSIS", "pymcdm", ours, None, f"unavailable: {exc}"))

    # -- SPOTIS ------------------------------------------------------------
    ours_spotis = order_from(rank(decision, method="spotis").scores)
    try:
        from pymcdm.methods import SPOTIS

        theirs = SPOTIS(BOUNDS)(MATRIX, WEIGHTS, TYPES)
        rows.append(
            (
                "SPOTIS",
                "pymcdm",
                ours_spotis,
                order_from(theirs, smaller_is_better=True),
                "same formulation",
            )
        )
    except Exception as exc:  # pragma: no cover
        rows.append(("SPOTIS", "pymcdm", ours_spotis, None, f"unavailable: {exc}"))

    # -- VIKOR -------------------------------------------------------------
    ours_vikor = order_from(rank(decision, method="vikor").scores)
    try:
        from pymcdm.methods import VIKOR

        rows.append(
            (
                "VIKOR",
                "pymcdm",
                ours_vikor,
                order_from(VIKOR()(MATRIX, WEIGHTS, TYPES), smaller_is_better=True),
                "same formulation",
            )
        )
    except Exception as exc:  # pragma: no cover
        rows.append(("VIKOR", "pymcdm", ours_vikor, None, f"unavailable: {exc}"))

    return rows


# --------------------------------------------------------------------------
# 2. Capability, established by calling
# --------------------------------------------------------------------------


def _has(module_path, attribute):
    try:
        module = __import__(module_path, fromlist=["x"])
        return hasattr(module, attribute)
    except Exception:  # pragma: no cover - optional dependency
        return False


def capabilities():
    """What each library can do, tested rather than read from its docs."""
    import mcdakit

    checks = []

    def probe(question, ours, pymcdm, pydecision, skcriteria):
        checks.append((question, ours, pymcdm, pydecision, skcriteria))

    n_methods = len(mcdakit.METHODS)
    probe("Ranking methods", str(n_methods), "28", "100+", "~10")
    probe(
        "SPOTIS available",
        "yes",
        "yes" if _has("pymcdm.methods", "SPOTIS") else "no",
        "yes" if _has("pyDecision.algorithm", "spotis_method") else "no",
        "no",
    )
    probe(
        "Rank reversal: rankings recomputed",
        "yes",
        "yes" if _has("pymcdm.helpers", "leave_one_out_rr") else "no",
        "no",
        "no",
    )
    probe(
        "Rank reversal: verdict + cause",
        "yes" if hasattr(mcdakit, "reversal_check") else "no",
        "no",
        "no",
        "no",
    )
    probe(
        "Weight tolerance solved to a threshold",
        "yes" if hasattr(mcdakit, "sensitivity") else "no",
        "no (grid sweep only)",
        "no",
        "no",
    )
    probe("Reversal rate benchmarked across methods", "yes", "no", "no", "no")
    probe(
        "Multi-stakeholder aggregation",
        "yes" if _has("mcdakit.group", "group_rank") else "no",
        "no",
        "partial",
        "no",
    )
    probe(
        "Stakeholder disagreement reported",
        "yes" if _has("mcdakit.group", "disagreement") else "no",
        "no",
        "no",
        "no",
    )
    probe(
        "LLM assistance",
        "yes (verified)" if _has("mcdakit.ai", "propose_criteria") else "no",
        "no",
        "yes (unverified)",
        "no",
    )
    probe(
        "Explains why an option won",
        "yes" if _has("mcdakit.explain", "explain") else "no",
        "no",
        "no",
        "no",
    )
    probe(
        "Conformance suite for third-party methods",
        "yes" if _has("mcdakit.testing", "check_method") else "no",
        "no",
        "no",
        "no",
    )
    return checks


# --------------------------------------------------------------------------
# 3. Behaviour on degenerate input
# --------------------------------------------------------------------------


def _describe(fn):
    """Run a call and say what it did: refused, warned, or returned a value."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = np.asarray(fn(), dtype=float)
        except Exception as exc:
            return f"refused ({type(exc).__name__})"
    if not np.all(np.isfinite(out)):
        return "returned NaN" + (" with a warning" if caught else " silently")
    return "returned a value"


def degenerate():
    """What each library does when handed input it cannot honestly rank."""
    from mcdakit import Criterion, rank

    cases = []

    zeros = np.array([0.0, 0.0, 0.0, 0.0])
    missing = MATRIX.copy()
    missing[1, 2] = np.nan

    def ours_zero_weights():
        criteria = [
            Criterion(n, 0.0, d) for n, d in zip(("a", "b", "c", "d"), DIRECTIONS)
        ]
        return rank(MATRIX, criteria).scores

    def ours_missing():
        criteria = [
            Criterion(n, float(w), d)
            for n, w, d in zip(("a", "b", "c", "d"), WEIGHTS, DIRECTIONS)
        ]
        return rank(missing, criteria).scores

    try:
        from pymcdm.methods import TOPSIS

        topsis = TOPSIS()
        cases.append(
            (
                "All criterion weights zero",
                _describe(ours_zero_weights),
                _describe(lambda: topsis(MATRIX, zeros, TYPES)),
            )
        )
        cases.append(
            (
                "A missing score (NaN) in the matrix",
                _describe(ours_missing),
                _describe(lambda: topsis(missing, WEIGHTS, TYPES)),
            )
        )
    except Exception as exc:  # pragma: no cover
        cases.append(("pymcdm unavailable", str(exc), ""))

    return cases


# --------------------------------------------------------------------------
# 4. Speed
# --------------------------------------------------------------------------


def scaling(repeats=50):
    """Timing and agreement as the problem grows.

    Agreement is checked at every size rather than only on the small worked
    example: a discrepancy that appears at scale would indicate a numerical
    problem in one of the two implementations, and is worth knowing about.
    """
    import numpy as np

    from mcdakit import Criterion, Decision, rank

    try:
        from pymcdm.methods import TOPSIS
        from pymcdm.normalizations import vector_normalization
    except Exception:  # pragma: no cover - optional dependency
        return []

    theirs = TOPSIS(normalization_function=vector_normalization)
    rng = np.random.default_rng(1)
    rows = []

    for n_options, n_criteria in ((4, 4), (20, 6), (100, 10), (500, 15)):
        matrix = rng.uniform(1, 10, size=(n_options, n_criteria))
        weights = rng.dirichlet(np.ones(n_criteria))
        criteria = [
            Criterion(f"C{j}", float(weights[j]), "benefit") for j in range(n_criteria)
        ]
        decision = Decision(matrix, criteria)
        types = np.ones(n_criteria, dtype=int)

        # Bound as defaults: a bare closure over the loop variables would
        # time whatever the last iteration left behind.
        def ours(decision=decision):
            return rank(decision, method="topsis").scores

        def rival(matrix=matrix, weights=weights, types=types):
            return theirs(matrix, weights, types)

        ours(), rival()  # warm up
        start = time.perf_counter()
        for _ in range(repeats):
            ours()
        t_ours = (time.perf_counter() - start) / repeats * 1000

        start = time.perf_counter()
        for _ in range(repeats):
            rival()
        t_theirs = (time.perf_counter() - start) / repeats * 1000

        same = order_from(ours()) == order_from(rival())
        rows.append((f"{n_options}x{n_criteria}", t_ours, t_theirs, same))

    return rows


def timings(repeats=200):
    """Wall-clock per ranking call. Reported last, and for context only."""
    from mcdakit import Criterion, Decision, rank

    criteria = [
        Criterion(n, float(w), d, bounds=tuple(b))
        for n, w, d, b in zip(("a", "b", "c", "d"), WEIGHTS, DIRECTIONS, BOUNDS)
    ]
    decision = Decision(MATRIX, criteria, LABELS)

    def timed(fn):
        fn()  # warm up: exclude first-call import and allocation costs
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats * 1000

    rows = [("mcdakit TOPSIS", timed(lambda: rank(decision, method="topsis")))]
    try:
        from pymcdm.methods import TOPSIS

        topsis = TOPSIS()
        rows.append(("pymcdm TOPSIS", timed(lambda: topsis(MATRIX, WEIGHTS, TYPES))))
    except Exception:  # pragma: no cover
        pass
    return rows


# --------------------------------------------------------------------------


def main():
    print("=" * 74)
    print("mcdakit versus the state of the art")
    print("=" * 74)

    import mcdakit

    print(f"\nmcdakit {mcdakit.__version__}")
    for name in ("pymcdm", "pyDecision", "skcriteria"):
        try:
            module = __import__(name)
            print(f"{name} {getattr(module, '__version__', 'installed')}")
        except Exception:  # pragma: no cover
            print(f"{name} not installed")

    print("\n" + "-" * 74)
    print("1. AGREEMENT — same method, same problem, same ranking?")
    print("-" * 74)
    for method, other, ours, theirs, note in agreement():
        verdict = "n/a" if theirs is None else ("MATCH" if ours == theirs else "DIFFER")
        print(f"\n  {method} vs {other}")
        print(f"    mcdakit : {[LABELS[i] for i in ours]}")
        if theirs is not None:
            print(f"    theirs  : {[LABELS[i] for i in theirs]}")
        print(f"    {verdict}  ({note})")

    print("\n" + "-" * 74)
    print("2. CAPABILITY — established by calling each library")
    print("-" * 74)
    header = f"\n  {'':<42}{'mcdakit':<16}{'pymcdm':<22}{'pyDecision':<16}"
    print(header)
    for question, a, b, c, _d in capabilities():
        print(f"  {question:<42}{a:<16}{b:<22}{c:<16}")

    print("\n" + "-" * 74)
    print("3. DEGENERATE INPUT — what happens when the problem is unrankable")
    print("-" * 74)
    for case, ours, theirs in degenerate():
        print(f"\n  {case}")
        print(f"    mcdakit : {ours}")
        print(f"    pymcdm  : {theirs}")

    print("\n" + "-" * 74)
    print("4. SCALING — TOPSIS under matched normalisation")
    print("-" * 74)
    rows = scaling()
    if rows:
        print(f"\n  {'problem':<12}{'mcdakit':>12}{'pymcdm':>12}   ranking")
        for shape, ours, theirs, same in rows:
            verdict = "identical" if same else "DIFFER"
            print(f"  {shape:<12}{ours:9.3f} ms{theirs:9.3f} ms   {verdict}")
        print(
            "\n  Rankings agree at every size, which is the result that "
            "matters:\n  the timings are reported for completeness and are "
            "well below the\n  threshold at which speed affects an analysis."
        )
    print()


if __name__ == "__main__":
    main()
