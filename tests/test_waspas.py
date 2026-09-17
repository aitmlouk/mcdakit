"""WASPAS, validated against two agreeing implementations and its definition.

The method blends a weighted sum with a weighted product. The two halves fail
differently — the sum lets a good score offset a terrible one, the product
does not — so the tests assert that blending actually happens rather than
checking only the default case.
"""

import numpy as np
import pytest

from mcdakit import Criterion, Decision, McdaError, rank
from mcdakit.methods.waspas import waspas

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

#: pymcdm 1.4.0 and pyrepo-mcda 0.1.15 both produce these, independently.
REFERENCE = np.array([0.820029, 0.823006, 0.831801, 0.544038])


def published(matrix, weights, directions, lambda_=0.5):
    """WASPAS from Zavadskas et al. (2012), written out independently."""
    matrix = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    is_cost = np.asarray(directions) == "cost"
    normalised = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        normalised[:, j] = (
            column.min() / column if is_cost[j] else column / column.max()
        )
    wsm = (normalised * w).sum(axis=1)
    wpm = np.prod(normalised**w, axis=1)
    return lambda_ * wsm + (1.0 - lambda_) * wpm


class TestAgainstIndependentSources:
    def test_it_matches_two_other_implementations(self):
        """pymcdm and pyrepo-mcda agree with each other here, which makes
        their shared value usable as ground truth."""
        assert waspas(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(REFERENCE, abs=1e-6)

    def test_it_matches_the_published_formula(self):
        assert waspas(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(
            published(MATRIX, WEIGHTS, DIRECTIONS), abs=1e-12
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_matches_on_random_problems(self, seed):
        rng = np.random.default_rng(seed)
        n_options = int(rng.integers(3, 8))
        n_criteria = int(rng.integers(3, 6))
        matrix = rng.uniform(0.5, 10, size=(n_options, n_criteria))
        weights = rng.dirichlet(np.ones(n_criteria))
        directions = ["cost" if j % 2 else "benefit" for j in range(n_criteria)]
        assert waspas(matrix, weights, directions) == pytest.approx(
            published(matrix, weights, directions), abs=1e-12
        )


class TestTheBlend:
    """lambda_ must actually select between the two models."""

    def test_lambda_one_is_the_weighted_sum(self):
        w = WEIGHTS / WEIGHTS.sum()
        is_cost = np.asarray(DIRECTIONS) == "cost"
        normalised = np.where(
            is_cost, MATRIX.min(axis=0) / MATRIX, MATRIX / MATRIX.max(axis=0)
        )
        assert waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=1.0) == pytest.approx(
            (normalised * w).sum(axis=1), abs=1e-12
        )

    def test_lambda_zero_is_the_weighted_product(self):
        w = WEIGHTS / WEIGHTS.sum()
        is_cost = np.asarray(DIRECTIONS) == "cost"
        normalised = np.where(
            is_cost, MATRIX.min(axis=0) / MATRIX, MATRIX / MATRIX.max(axis=0)
        )
        assert waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=0.0) == pytest.approx(
            np.prod(normalised**w, axis=1), abs=1e-12
        )

    def test_the_default_sits_between_the_two(self):
        low = waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=0.0)
        mid = waspas(MATRIX, WEIGHTS, DIRECTIONS)
        high = waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=1.0)
        assert np.all(mid >= np.minimum(low, high) - 1e-12)
        assert np.all(mid <= np.maximum(low, high) + 1e-12)

    def test_the_two_halves_can_choose_different_winners(self):
        """The reason for having both, and for lambda_ mattering.

        The sum is compensatory: a strong score can offset a weak one. The
        product is not, because a low value drags the whole product down. On
        this problem the second alternative wins on the sum despite scoring
        1.24 on its middle criterion, and loses on the product because of it.
        Found by search rather than constructed, since the effect is less
        common than the textbook description suggests --- the normalisation
        already penalises a weak column in both halves.
        """
        matrix = np.array(
            [
                [4.205, 7.371, 7.140],
                [9.327, 1.238, 7.317],
                [9.281, 9.682, 0.246],
            ]
        )
        weights = np.array([0.317, 0.242, 0.441])
        directions = ["benefit"] * 3

        by_sum = waspas(matrix, weights, directions, lambda_=1.0)
        by_product = waspas(matrix, weights, directions, lambda_=0.0)

        assert int(np.argmax(by_sum)) == 1
        assert int(np.argmax(by_product)) == 0, (
            "the product should reject the alternative with a weak criterion"
        )

    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_a_lambda_outside_the_unit_interval_is_refused(self, bad):
        with pytest.raises(McdaError, match="between 0 and 1"):
            waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=bad)


class TestDegenerateInput:
    def test_a_non_positive_value_is_refused_with_a_reason(self):
        """Zero divides in the cost normalisation and collapses the product."""
        for bad in (0.0, -1.0):
            matrix = MATRIX.copy()
            matrix[0, 0] = bad
            with pytest.raises(McdaError, match="strictly positive"):
                waspas(matrix, WEIGHTS, DIRECTIONS)

    def test_all_zero_weights_return_zeros(self):
        assert waspas(MATRIX, np.zeros(4), DIRECTIONS) == pytest.approx(np.zeros(4))

    def test_a_flat_criterion_is_handled(self):
        matrix = np.array([[5.0, 1.0], [5.0, 9.0]])
        scores = waspas(matrix, np.array([0.5, 0.5]), ["benefit", "benefit"])
        assert np.all(np.isfinite(scores))


class TestThroughTheApi:
    def _criteria(self):
        return [
            Criterion(n, float(w), d)
            for n, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]

    def test_it_is_registered(self):
        from mcdakit import METHODS

        assert "waspas" in METHODS

    def test_rank_reproduces_the_reference_scores(self):
        result = rank(Decision(MATRIX, self._criteria(), LABELS), method="waspas")
        assert result.scores == pytest.approx(REFERENCE, abs=1e-6)

    def test_lambda_reaches_the_method_through_rank(self):
        decision = Decision(MATRIX, self._criteria(), LABELS)
        assert not np.allclose(
            rank(decision, method="waspas", lambda_=0.0).scores,
            rank(decision, method="waspas", lambda_=1.0).scores,
        )

    def test_it_takes_the_matrix_as_measured(self):
        from mcdakit.methods import Waspas
        from mcdakit.methods.base import Wants

        assert Waspas.wants is Wants.RAW

    def test_it_passes_the_conformance_suite(self):
        from mcdakit.methods import Waspas
        from mcdakit.testing import conformance_report

        failures = {k: v for k, v in conformance_report(Waspas()).items() if v}
        assert not failures, failures

    def test_sensitivity_works_on_it(self):
        from mcdakit import sensitivity

        report = sensitivity(
            rank(Decision(MATRIX, self._criteria(), LABELS), method="waspas")
        )
        assert report["level"] in {"fragile", "moderate", "robust", "immovable"}
