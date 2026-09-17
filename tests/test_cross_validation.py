"""Cross-check against `pymcdm`, the reference implementation.

Agreeing with an independent implementation is worth more than agreeing with
ourselves. Where the two differ, the difference is documented here with the
reason — an unexplained disagreement would be a bug.

Skipped when pymcdm is absent; it is a dev-only extra (`pip install
mcdakit[compare]`) and must never become a runtime dependency.
"""

import numpy as np
import pytest

from mcdakit import Criterion, rank

pymcdm = pytest.importorskip("pymcdm", reason="dev-only cross-check")

from pymcdm.methods import SPOTIS, TOPSIS  # noqa: E402
from pymcdm.normalizations import (  # noqa: E402
    minmax_normalization,
    vector_normalization,
)

# pymcdm encodes criterion types as 1 for profit/benefit and -1 for cost.
# Verified empirically against its SPOTIS, not assumed from the docs.
PYMCDM_BENEFIT, PYMCDM_COST = 1, -1

MATRIX = np.array(
    [
        [10.5, -3.1, 1.7, 3.4],
        [-4.7, 0.0, 3.4, 5.6],
        [8.1, 0.3, 1.3, 2.1],
        [3.2, 7.3, -5.3, 1.9],
    ]
)
WEIGHTS = np.array([0.2, 0.3, 0.4, 0.1])
BOUNDS = np.array([[-5.0, 12], [-6, 10], [-8, 5], [-4, 6]])
DIRECTIONS = ["benefit", "cost", "benefit", "benefit"]
TYPES = np.array([PYMCDM_BENEFIT, PYMCDM_COST, PYMCDM_BENEFIT, PYMCDM_BENEFIT])


def _criteria(directions=None, bounds=True):
    directions = directions or DIRECTIONS
    return [
        Criterion(
            f"C{j + 1}",
            float(WEIGHTS[j]),
            directions[j],
            bounds=tuple(BOUNDS[j]) if bounds else None,
        )
        for j in range(len(WEIGHTS))
    ]


class TestSpotis:
    def test_scores_match_to_machine_precision(self):
        """SPOTIS is the differentiator, so it is the one that most needs an
        outside witness."""
        theirs = SPOTIS(BOUNDS)(MATRIX, WEIGHTS, TYPES)
        ours = rank(MATRIX, _criteria(), method="spotis").scores
        # We negate so higher is better; pymcdm reports the raw distance.
        assert np.allclose(-ours, theirs, atol=1e-12)

    def test_the_ranking_matches(self):
        theirs = SPOTIS(BOUNDS)(MATRIX, WEIGHTS, TYPES)
        ours = rank(MATRIX, _criteria(), method="spotis")
        assert list(np.argsort(theirs)) == [
            ours.decision.labels.index(label) for label in ours.order
        ]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_they_agree_on_random_problems(self, seed):
        rng = np.random.default_rng(seed)
        matrix = rng.uniform(0, 10, size=(6, 4))
        weights = rng.dirichlet(np.ones(4))
        bounds = np.array([[0.0, 10.0]] * 4)
        types = np.array([PYMCDM_BENEFIT, PYMCDM_COST, PYMCDM_BENEFIT, PYMCDM_COST])
        directions = ["benefit", "cost", "benefit", "cost"]

        criteria = [
            Criterion(f"C{j + 1}", float(weights[j]), directions[j], bounds=(0.0, 10.0))
            for j in range(4)
        ]
        theirs = SPOTIS(bounds)(matrix, weights, types)
        ours = rank(matrix, criteria, method="spotis").scores
        assert np.allclose(-ours, theirs, atol=1e-12)


class TestTopsis:
    """A documented, deliberate difference in normalisation.

    `mcdakit` uses **vector** normalisation — ``x / sqrt(sum(x^2))`` — which is
    Hwang and Yoon's original 1981 formulation. pymcdm's TOPSIS defaults to
    **min-max** normalisation instead. Neither is wrong; they are different
    modelling choices, and the choice genuinely moves the numbers.

    Told to use vector normalisation, pymcdm reproduces our scores to machine
    precision, which is what proves the difference is the normalisation and
    not an error in the method itself.
    """

    POSITIVE = np.abs(MATRIX) + 1.0
    ALL_BENEFIT = np.array([PYMCDM_BENEFIT] * 4)

    def test_we_match_pymcdm_under_the_same_normalisation(self):
        theirs = TOPSIS(normalization_function=vector_normalization)(
            self.POSITIVE, WEIGHTS, self.ALL_BENEFIT
        )
        ours = rank(
            self.POSITIVE, _criteria(["benefit"] * 4, bounds=False), method="topsis"
        ).scores
        assert np.allclose(ours, theirs, atol=1e-12)

    def test_pymcdms_default_differs_and_we_know_why(self):
        """Pin the disagreement so a future change to either side is noticed
        rather than quietly absorbed."""
        default = TOPSIS()(self.POSITIVE, WEIGHTS, self.ALL_BENEFIT)
        ours = rank(
            self.POSITIVE, _criteria(["benefit"] * 4, bounds=False), method="topsis"
        ).scores
        assert not np.allclose(ours, default, atol=1e-6)
        minmax = TOPSIS(normalization_function=minmax_normalization)(
            self.POSITIVE, WEIGHTS, self.ALL_BENEFIT
        )
        assert np.allclose(default, minmax, atol=1e-12), (
            "pymcdm's default is min-max normalisation"
        )

    def test_both_still_rank_a_dominant_option_first(self):
        """Whatever the normalisation, the one answer neither may get wrong."""
        dominant = self.POSITIVE.max(axis=0) + 1.0
        extended = np.vstack([self.POSITIVE, dominant])

        theirs = TOPSIS()(extended, WEIGHTS, np.array([PYMCDM_BENEFIT] * 4))
        ours = rank(
            extended, _criteria(["benefit"] * 4, bounds=False), method="topsis"
        ).scores
        assert int(np.argmax(theirs)) == int(np.argmax(ours)) == 4


class TestRankingsAgreeAtScale:
    """Agreement with the reference implementation, on problems large enough
    that a numerical discrepancy would have room to appear.

    The worked example above pins particular values; this pins the property
    that the two implementations order alternatives the same way as the
    problem grows. A divergence here would indicate a numerical defect in one
    of them, and is worth failing the build for.
    """

    @pytest.mark.parametrize(
        "shape", [(4, 4), (20, 6), (100, 10)], ids=lambda s: f"{s[0]}x{s[1]}"
    )
    def test_topsis_matches_under_the_same_normalisation(self, shape):
        from pymcdm.normalizations import vector_normalization

        from mcdakit import Criterion, Decision, rank

        n_options, n_criteria = shape
        rng = np.random.default_rng(1)
        matrix = rng.uniform(1, 10, size=(n_options, n_criteria))
        weights = rng.dirichlet(np.ones(n_criteria))

        criteria = [
            Criterion(f"C{j}", float(weights[j]), "benefit") for j in range(n_criteria)
        ]
        ours = rank(Decision(matrix, criteria), method="topsis").scores
        theirs = TOPSIS(normalization_function=vector_normalization)(
            matrix, weights, np.ones(n_criteria, dtype=int)
        )

        assert np.allclose(ours, theirs, atol=1e-10)
        assert list(np.argsort(-ours)) == list(np.argsort(-theirs))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_spotis_matches_on_mixed_direction_problems(self, seed):
        """Direction handling is where the two libraries could plausibly
        differ, since they encode it differently (strings versus +1/-1)."""
        from pymcdm.methods import SPOTIS

        from mcdakit import Criterion, Decision, rank

        rng = np.random.default_rng(seed)
        matrix = rng.uniform(0, 10, size=(8, 5))
        weights = rng.dirichlet(np.ones(5))
        directions = ["cost" if j % 2 else "benefit" for j in range(5)]
        bounds = np.array([[0.0, 10.0]] * 5)

        criteria = [
            Criterion(f"C{j}", float(weights[j]), directions[j], bounds=(0.0, 10.0))
            for j in range(5)
        ]
        ours = rank(Decision(matrix, criteria), method="spotis").scores
        types = np.array(
            [PYMCDM_COST if d == "cost" else PYMCDM_BENEFIT for d in directions]
        )
        theirs = SPOTIS(bounds)(matrix, weights, types)

        # We negate so that larger is better; pymcdm reports the distance.
        assert np.allclose(-ours, theirs, atol=1e-12)


class TestAgainstHandComputation:
    """The strongest evidence available: arithmetic verifiable on paper.

    Cross-library agreement shows two implementations concur; it does not
    show either is right. These tests compute the method from its published
    definition, independently of the package, and compare.
    """

    MATRIX = np.array(
        [
            [2.75, 7.0, 14.0, 8.0],
            [2.90, 8.5, 16.0, 8.0],
            [3.40, 9.0, 11.0, 7.0],
            [2.20, 3.0, 32.0, 2.0],
        ]
    )
    WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15])
    COST = [True, False, True, False]

    def _vikor_by_hand(self, v=0.5):
        """VIKOR from Opricovic and Tzeng's definition, written out in full."""
        n_options, n_criteria = self.MATRIX.shape
        S = np.zeros(n_options)
        R = np.zeros(n_options)
        for i in range(n_options):
            terms = []
            for j in range(n_criteria):
                column = self.MATRIX[:, j]
                best = column.min() if self.COST[j] else column.max()
                worst = column.max() if self.COST[j] else column.min()
                denominator = best - worst if best != worst else 1.0
                terms.append(self.WEIGHTS[j] * (best - self.MATRIX[i, j]) / denominator)
            S[i] = sum(terms)
            R[i] = max(terms)
        return v * (S - S.min()) / (S.max() - S.min()) + (1 - v) * (R - R.min()) / (
            R.max() - R.min()
        )

    def test_vikor_matches_the_textbook_formula(self):
        """Q values, not merely the ordering: an ordering can coincide by
        accident where four values to four decimal places cannot."""
        from mcdakit import Criterion, Decision, rank

        criteria = [
            Criterion(
                f"C{j}", float(self.WEIGHTS[j]), "cost" if self.COST[j] else "benefit"
            )
            for j in range(4)
        ]
        # The package returns -Q so that larger is better throughout.
        ours = -rank(Decision(self.MATRIX, criteria), method="vikor").scores
        assert ours == pytest.approx(self._vikor_by_hand(), abs=1e-9)

    def test_vikor_respects_the_v_parameter(self):
        from mcdakit import Criterion, Decision, rank

        criteria = [
            Criterion(
                f"C{j}", float(self.WEIGHTS[j]), "cost" if self.COST[j] else "benefit"
            )
            for j in range(4)
        ]
        decision = Decision(self.MATRIX, criteria)
        for v in (0.0, 0.25, 0.75, 1.0):
            ours = -rank(decision, method="vikor", v=v).scores
            assert ours == pytest.approx(self._vikor_by_hand(v), abs=1e-9)


class TestAgainstPyDecision:
    """A second independent implementation, for the methods pymcdm lacks.

    pyDecision returns tables of ``[option_number, value]`` sorted by value,
    so the option number in column zero is the only safe way to recover which
    score belongs to which alternative. Reading column one positionally
    produced a spurious VIKOR disagreement while this suite was being written.
    """

    MATRIX = TestAgainstHandComputation.MATRIX
    WEIGHTS = TestAgainstHandComputation.WEIGHTS
    TYPES = ["min", "max", "min", "max"]

    def _criteria(self):
        from mcdakit import Criterion

        return [
            Criterion(
                f"C{j}", float(self.WEIGHTS[j]), "cost" if t == "min" else "benefit"
            )
            for j, t in enumerate(self.TYPES)
        ]

    def _quiet(self, fn):
        import contextlib
        import io
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()

    def test_vikor_agrees_with_a_third_implementation(self):
        """Three independent implementations converging is materially
        stronger evidence than two."""
        pytest.importorskip("pyDecision", reason="dev-only cross-check")
        from pyDecision.algorithm import vikor_method

        from mcdakit import Decision, rank

        out = self._quiet(
            lambda: vikor_method(
                self.MATRIX,
                self.WEIGHTS,
                self.TYPES,
                graph=False,
                verbose=False,
            )
        )
        table = np.asarray(out[2], dtype=float)
        theirs = {int(i) - 1: q for i, q in table}

        ours = -rank(Decision(self.MATRIX, self._criteria()), method="vikor").scores
        for index, q in theirs.items():
            assert ours[index] == pytest.approx(q, abs=1e-9)
