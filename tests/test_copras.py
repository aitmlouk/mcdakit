"""COPRAS, validated against three independent sources.

The method's defining feature is that its second term is *inversely*
proportional to an alternative's cost total, so cheaper is better. An
implementation that adds the cost total instead still produces a plausible
ranking, which is why that property is asserted directly rather than inferred
from agreement with another library.
"""

import numpy as np
import pytest

from mcdakit import Criterion, Decision, McdaError, rank
from mcdakit.methods.copras import copras

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
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]


def published(matrix, weights, directions):
    """COPRAS written out from Zavadskas et al. (1994), independently.

    Kept deliberately separate from the implementation: a test that reuses the
    code under test proves only that it is self-consistent.
    """
    matrix = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    normalised = matrix / matrix.sum(axis=0)
    weighted = normalised * w
    is_cost = np.asarray(directions) == "cost"
    s_plus = weighted[:, ~is_cost].sum(axis=1)
    s_minus = weighted[:, is_cost].sum(axis=1)
    return s_plus + (s_minus.min() * s_minus.sum()) / (
        s_minus * np.sum(s_minus.min() / s_minus)
    )


class TestAgainstThePublishedFormula:
    def test_it_matches_the_definition(self):
        assert copras(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(
            published(MATRIX, WEIGHTS, DIRECTIONS), abs=1e-12
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_matches_on_random_problems(self, seed):
        rng = np.random.default_rng(seed)
        n_options = int(rng.integers(3, 8))
        n_criteria = int(rng.integers(3, 6))
        matrix = rng.uniform(1, 10, size=(n_options, n_criteria))
        weights = rng.dirichlet(np.ones(n_criteria))
        directions = ["cost" if j % 2 else "benefit" for j in range(n_criteria)]

        assert copras(matrix, weights, directions) == pytest.approx(
            published(matrix, weights, directions), abs=1e-12
        )


class TestTheDefiningProperty:
    """Cheaper must score higher. This is what separates COPRAS from a
    weighted sum, and an implementation can get it backwards while still
    looking reasonable."""

    def test_a_cheaper_alternative_scores_higher(self):
        matrix = np.array([[10.0, 5.0], [20.0, 5.0]])
        scores = copras(matrix, np.array([0.5, 0.5]), ["cost", "benefit"])
        assert scores[0] > scores[1], (
            "the cheaper alternative must score higher; adding the cost total "
            "rather than inverting it would reverse this"
        )

    def test_raising_a_cost_lowers_the_score(self):
        base = copras(MATRIX, WEIGHTS, DIRECTIONS)
        dearer = MATRIX.copy()
        dearer[0, 0] *= 2.0
        assert copras(dearer, WEIGHTS, DIRECTIONS)[0] < base[0]

    def test_raising_a_benefit_raises_the_score(self):
        base = copras(MATRIX, WEIGHTS, DIRECTIONS)
        better = MATRIX.copy()
        better[0, 1] *= 1.5
        assert copras(better, WEIGHTS, DIRECTIONS)[0] > base[0]


class TestDegenerateInput:
    def test_with_no_cost_criterion_it_reduces_to_a_weighted_sum(self):
        """The proportional term is undefined without costs; falling back to
        the benefit total is the only sensible reading."""
        benefit_only = ["benefit"] * 4
        scores = copras(MATRIX, WEIGHTS, benefit_only)
        normalised = MATRIX / MATRIX.sum(axis=0)
        expected = (normalised * (WEIGHTS / WEIGHTS.sum())).sum(axis=1)
        assert scores == pytest.approx(expected, abs=1e-12)

    def test_a_zero_cost_total_does_not_divide_by_zero(self):
        matrix = np.array([[0.0, 5.0], [4.0, 5.0], [8.0, 5.0]])
        scores = copras(matrix, np.array([0.5, 0.5]), ["cost", "benefit"])
        assert np.all(np.isfinite(scores))
        assert scores[0] == max(scores), "costing nothing should rank first"

    def test_all_zero_weights_return_zeros(self):
        scores = copras(MATRIX, np.zeros(4), DIRECTIONS)
        assert scores == pytest.approx(np.zeros(4))

    def test_a_flat_column_does_not_divide_by_zero(self):
        matrix = np.array([[0.0, 5.0], [0.0, 9.0]])
        assert np.all(
            np.isfinite(copras(matrix, np.array([0.5, 0.5]), ["cost", "benefit"]))
        )


class TestThroughTheApi:
    def test_it_is_registered(self):
        from mcdakit import METHODS

        assert "copras" in METHODS

    def test_rank_produces_the_expected_order(self):
        criteria = [
            Criterion(n, float(w), d)
            for n, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]
        result = rank(Decision(MATRIX, criteria, LABELS), method="copras")
        assert result.order == ["Option 2", "Option 1", "Option 3", "Option 4"]

    def test_it_takes_the_matrix_as_measured(self):
        """COPRAS resolves direction itself, so pre-orienting would apply the
        correction twice and rank costs backwards."""
        from mcdakit.methods import Copras
        from mcdakit.methods.base import Wants

        assert Copras.wants is Wants.RAW

    def test_it_passes_the_conformance_suite(self):
        from mcdakit.methods import Copras
        from mcdakit.testing import conformance_report

        failures = {k: v for k, v in conformance_report(Copras()).items() if v}
        assert not failures, failures

    def test_sensitivity_works_on_it(self):
        criteria = [
            Criterion(n, float(w), d)
            for n, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]
        from mcdakit import sensitivity

        report = sensitivity(rank(Decision(MATRIX, criteria, LABELS), method="copras"))
        assert report["level"] in {"fragile", "moderate", "robust", "immovable"}

    def test_a_negative_value_is_refused_with_a_reason(self):
        """Sum-normalisation is defined for ratio-scale data. A negative value
        makes the column total meaningless, and the resulting NaN would sort
        unpredictably rather than fail visibly."""
        criteria = [Criterion("A", 1.0, "cost"), Criterion("B", 1.0)]
        with pytest.raises(McdaError, match="non-negative"):
            rank([[-1.0, 2.0], [1.0, 2.0]], criteria, method="copras")
