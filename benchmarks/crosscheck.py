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

    pip install -e ".[compare]"

Every library that implements a method is compared against it. Where two
disagree, the difference is either a defect or a documented modelling choice;
``EXPECTED_DIFFERENCES`` records the attributed ones so a known choice is not
re-reported as a fresh discrepancy, and any unattributed difference fails the
run. ``check_orientation_conventions`` backs the attribution up: it re-runs
the problem with every criterion declared a benefit, where no cost transform
fires anywhere, and requires the libraries to agree to machine precision.
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


def _pyrepo_copras():
    from pyrepo_mcda.mcda_methods import COPRAS

    with quiet():
        return order(COPRAS()(MATRIX, WEIGHTS, TYPES))


def _pymcdm_copras():
    from pymcdm.methods import COPRAS

    return order(COPRAS()(MATRIX, WEIGHTS, TYPES))


def _pymcdm_waspas():
    from pymcdm.methods import WASPAS

    return order(WASPAS()(MATRIX, WEIGHTS, TYPES))


def _pyrepo_waspas():
    from pyrepo_mcda.mcda_methods import WASPAS

    with quiet():
        return order(WASPAS()(MATRIX, WEIGHTS, TYPES))


def _pymcdm_wpm():
    """pymcdm's WPM defaults to sum normalisation where mcdakit uses linear.

    Both are standard. Forced onto the same normalisation the two agree to
    1e-12; left on their defaults they still agree on the order, which is
    what this harness compares.
    """
    from pymcdm import normalizations as nz
    from pymcdm.methods import WPM

    return order(
        WPM(normalization_function=nz.linear_normalization)(MATRIX, WEIGHTS, TYPES)
    )


def _pyrepo_topsis():
    """pyrepo's TOPSIS defaults to min-max; forced onto vector normalisation it
    still differs, because it treats a cost criterion as ``1 - x/||x||``
    (complement after normalising) where mcdakit mirrors the interval
    (``max + min - x``) before normalising. Both are published variants. On an
    all-benefit matrix, where neither transform fires, the two agree to 1e-12
    — see ``check_orientation_conventions``.
    """
    from pyrepo_mcda import normalizations as nz
    from pyrepo_mcda.mcda_methods import TOPSIS

    with quiet():
        return order(
            TOPSIS(normalization_method=nz.vector_normalization)(MATRIX, WEIGHTS, TYPES)
        )


def _pyrepo_saw():
    """Same cost convention as pyDecision: ``min/x`` rather than the interval
    mirror. Agrees with mcdakit to 1e-12 on all-benefit data."""
    from pyrepo_mcda.mcda_methods import SAW

    with quiet():
        return order(SAW()(MATRIX, WEIGHTS, TYPES))


def _skcriteria_topsis():
    """scikit-criteria inverts cost criteria before scaling. Agrees with
    mcdakit to 1e-9 on all-benefit data."""
    import skcriteria as sk
    from skcriteria.agg.similarity import TOPSIS
    from skcriteria.preprocessing.scalers import VectorScaler

    dm = sk.mkdm(
        MATRIX, [min if d == "cost" else max for d in DIRECTIONS], weights=WEIGHTS
    )
    with quiet():
        result = TOPSIS().evaluate(VectorScaler(target="matrix").transform(dm))
    return tuple(np.argsort(result.rank_))


def _pymcdm_named(name):
    """A pymcdm method invoked by name, for the many that share a signature."""

    def check():
        from pymcdm import methods as pm

        return order(
            np.asarray(getattr(pm, name)()(MATRIX, WEIGHTS, TYPES), dtype=float)
        )

    return check


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
        "pyrepo-mcda": _pyrepo_topsis,
        "scikit-criteria": _skcriteria_topsis,
    },
    "spotis": {"pymcdm": _pymcdm_spotis},
    "vikor": {"pymcdm": _pymcdm_vikor, "pyDecision": _pydecision_vikor},
    "promethee": {"pymcdm (usual)": _pymcdm_promethee},
    "saw": {"pyDecision": _pydecision_saw, "pyrepo-mcda": _pyrepo_saw},
    "waspas": {
        "pymcdm": _pymcdm_waspas,
        "pyrepo-mcda": _pyrepo_waspas,
    },
    "wpm": {"pymcdm (linear norm)": _pymcdm_wpm},
    "aras": {"pymcdm": _pymcdm_named("ARAS")},
    "cocoso": {"pymcdm": _pymcdm_named("COCOSO")},
    "codas": {"pymcdm": _pymcdm_named("CODAS")},
    "edas": {"pymcdm": _pymcdm_named("EDAS")},
    "mabac": {"pymcdm": _pymcdm_named("MABAC")},
    "marcos": {"pymcdm": _pymcdm_named("MARCOS")},
    "copras": {
        "pyrepo-mcda": _pyrepo_copras,
        "pymcdm": _pymcdm_copras,
    },
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
    ("copras", "pymcdm"): (
        "pymcdm's COPRAS computes Sp + (min(Sm)*Sm)/(Sm*(min(Sm)/Sm)), whose "
        "second term simplifies identically to Sm, so it adds the cost total "
        "rather than inverting it. pyrepo-mcda, pyDecision and the formula of "
        "Zavadskas et al. (1994) all invert it, and mcdakit follows those "
        "three."
    ),
    ("topsis", "pyrepo-mcda"): (
        "pyrepo treats a cost criterion as 1 - x/||x||, taking the complement "
        "after normalising; mcdakit mirrors the interval (max + min - x) "
        "before normalising. Both appear in the literature. On an all-benefit "
        "matrix the two agree to 1e-12, which isolates the difference to the "
        "cost transform alone."
    ),
    ("topsis", "scikit-criteria"): (
        "scikit-criteria inverts cost criteria before scaling rather than "
        "mirroring the interval. It agrees with mcdakit to 1e-9 on an "
        "all-benefit matrix."
    ),
    ("saw", "pyrepo-mcda"): (
        "pyrepo uses the same cost convention as pyDecision (min/x) rather "
        "than the interval mirror. Agrees with mcdakit to 1e-12 on an "
        "all-benefit matrix."
    ),
    ("topsis", "pymcdm (default min-max)"): (
        "pymcdm defaults to min-max normalisation; mcdakit follows Hwang and "
        "Yoon's vector normalisation. Matching the normalisation makes them "
        "agree, as the row above shows."
    ),
}


#: Saaty's car hierarchy. AHP takes comparison matrices rather than the shared
#: decision matrix, so it is cross-checked separately from CHECKS.
AHP_CRITERIA = np.array([[1, 1 / 2, 3], [2, 1, 4], [1 / 3, 1 / 4, 1]], dtype=float)
AHP_ALTERNATIVES = [
    np.array([[1, 1 / 4, 4], [4, 1, 4], [1 / 4, 1 / 4, 1]], dtype=float),
    np.array([[1, 2, 5], [1 / 2, 1, 3], [1 / 5, 1 / 3, 1]], dtype=float),
    np.array([[1, 1 / 3, 1 / 2], [3, 1, 2], [2, 1 / 2, 1]], dtype=float),
]


def check_ahp() -> bool:
    """Compare full AHP against pyrepo-mcda's ``AHP._classic_ahp``.

    The public ``AHP`` of pyrepo-mcda is a weighted sum over a numeric matrix,
    and pyDecision's ``ahp_method`` is the weighting step alone; neither
    compares alternatives pairwise. The private ``_classic_ahp`` does, so it is
    the only independent implementation to check against. Priorities are
    compared numerically rather than by rank, the agreement being exact.
    """
    from mcdakit import ahp_rank

    print(f"\n{'-' * 74}\nahp (full hierarchy)\n{'-' * 74}")
    mine = ahp_rank(AHP_CRITERIA, AHP_ALTERNATIVES)
    print(f"  mcdakit              {np.round(mine.scores, 6)}")

    try:
        from pyrepo_mcda.mcda_methods import AHP
    except Exception as exc:
        print(f"  {'pyrepo-mcda':<21}unavailable: {type(exc).__name__}: {exc}")
        return True

    rival = AHP()
    with quiet():
        weights = rival._geometric_mean(AHP_CRITERIA)
        theirs = rival._classic_ahp(AHP_ALTERNATIVES, weights, rival._geometric_mean)
    agrees = bool(np.allclose(theirs, mine.scores, atol=1e-12))
    verdict = "MATCH " if agrees else "DIFFER"
    print(f"  {'pyrepo-mcda':<21}{np.round(theirs, 6)}  {verdict}")
    return agrees


def check_orientation_conventions() -> bool:
    """Prove that the TOPSIS and SAW disagreements are cost handling alone.

    Libraries differ in how they turn a cost criterion into a benefit one:
    mcdakit mirrors the interval (``max + min - x``), pyrepo takes a
    complement or a ratio, scikit-criteria inverts. Those are modelling
    choices, not defects — but saying so is only credible if the underlying
    algorithms agree once the choice is removed.

    So run the same problem with every criterion declared a benefit, where no
    cost transform fires in any library. What remains is the core algorithm,
    and it must agree to machine precision.
    """
    from mcdakit import Criterion, Decision, rank

    benefit = [
        Criterion(n, float(w), "benefit")
        for n, w in zip(("Price", "Quality", "Lead time", "Support"), WEIGHTS)
    ]
    all_benefit = np.ones(4, dtype=int)

    print(f"\n{'-' * 74}\nsame problem, all criteria declared benefit\n{'-' * 74}")
    print("  (no cost transform fires, so only the core algorithm is compared)")

    ok = True
    for method, get_theirs in (
        ("topsis", _all_benefit_topsis),
        ("saw", _all_benefit_saw),
    ):
        mine = rank(Decision(MATRIX, benefit, LABELS), method=method).scores
        for label, fn in get_theirs(all_benefit):
            try:
                theirs = fn()
            except Exception as exc:
                print(f"  {method:<8} {label:<18} unavailable: {type(exc).__name__}")
                continue
            agrees = bool(np.allclose(theirs, mine, atol=1e-9))
            print(
                f"  {method:<8} {label:<18} {'MATCH ' if agrees else 'DIFFER'}"
                f"  max|delta| = {np.abs(np.asarray(theirs) - mine).max():.2e}"
            )
            ok = ok and agrees
    return ok


def _all_benefit_topsis(types):
    def pyrepo():
        from pyrepo_mcda import normalizations as nz
        from pyrepo_mcda.mcda_methods import TOPSIS

        with quiet():
            return np.asarray(
                TOPSIS(normalization_method=nz.vector_normalization)(
                    MATRIX, WEIGHTS, types
                ),
                dtype=float,
            )

    def skcrit():
        import skcriteria as sk
        from skcriteria.agg.similarity import TOPSIS
        from skcriteria.preprocessing.scalers import VectorScaler

        dm = sk.mkdm(MATRIX, [max] * 4, weights=WEIGHTS)
        with quiet():
            return np.asarray(
                TOPSIS()
                .evaluate(VectorScaler(target="matrix").transform(dm))
                .e_.similarity,
                dtype=float,
            )

    def pymcdm_vec():
        from pymcdm.methods import TOPSIS
        from pymcdm.normalizations import vector_normalization

        return np.asarray(
            TOPSIS(normalization_function=vector_normalization)(MATRIX, WEIGHTS, types),
            dtype=float,
        )

    return (
        ("pyrepo-mcda", pyrepo),
        ("scikit-criteria", skcrit),
        ("pymcdm", pymcdm_vec),
    )


def _all_benefit_saw(types):
    def pyrepo():
        from pyrepo_mcda.mcda_methods import SAW

        with quiet():
            return np.asarray(SAW()(MATRIX, WEIGHTS, types), dtype=float)

    return (("pyrepo-mcda", pyrepo),)


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

    if not check_ahp():
        unexplained.append(("ahp", "pyrepo-mcda"))

    if not check_orientation_conventions():
        unexplained.append(("topsis/saw", "all-benefit isolation"))

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
