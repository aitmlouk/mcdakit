"""The weighted product model, validated against its definition and pymcdm.

WPM is WASPAS at lambda zero, and is implemented as that call. The tests here
therefore concentrate on the property that makes it a distinct method worth
exposing — that it refuses to let a strength pay for a weakness — rather than
re-testing the normalisation, which ``test_waspas.py`` already covers.
"""

import numpy as np
import pytest

from mcdakit import Criterion, Decision, McdaError, rank
from mcdakit.methods.waspas import waspas, wpm

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

#: pymcdm 1.4.0 produces these when told to use the same linear normalisation
#: mcdakit uses; its own default is sum normalisation, which rescales the
#: values without changing their order.
REFERENCE = np.array([0.81847008, 0.81895213, 0.82352865, 0.49849319])


def published(matrix, weights, directions):
    """WPM written out independently: normalise, raise to the weight, multiply."""
    matrix = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    is_cost = np.asarray(directions) == "cost"
    scores = np.ones(matrix.shape[0])
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        r = column.min() / column if is_cost[j] else column / column.max()
        scores = scores * r ** w[j]
    return scores


class TestAgainstIndependentSources:
    def test_it_matches_pymcdm_under_the_same_normalisation(self):
        assert wpm(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(REFERENCE, abs=1e-8)

    def test_it_matches_the_published_formula(self):
        assert wpm(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(
            published(MATRIX, WEIGHTS, DIRECTIONS), abs=1e-12
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_it_matches_the_formula_on_random_problems(self, seed):
        rng = np.random.default_rng(seed)
        rows, columns = int(rng.integers(3, 8)), int(rng.integers(2, 6))
        matrix = rng.uniform(0.5, 50.0, size=(rows, columns))
        weights = rng.uniform(0.1, 1.0, size=columns)
        directions = list(rng.choice(["benefit", "cost"], size=columns))
        assert wpm(matrix, weights, directions) == pytest.approx(
            published(matrix, weights, directions), abs=1e-12
        )


class TestItIsWaspasAtLambdaZero:
    """Implemented as that call, so the two cannot drift apart."""

    def test_it_equals_the_blend_at_lambda_zero(self):
        assert wpm(MATRIX, WEIGHTS, DIRECTIONS) == pytest.approx(
            waspas(MATRIX, WEIGHTS, DIRECTIONS, lambda_=0.0), abs=1e-15
        )

    def test_it_differs_from_the_default_blend(self):
        """Otherwise exposing it separately would be pointless."""
        assert not np.allclose(
            wpm(MATRIX, WEIGHTS, DIRECTIONS), waspas(MATRIX, WEIGHTS, DIRECTIONS)
        )


class TestItDoesNotCompensate:
    """The reason to choose WPM over a weighted sum."""

    #: Three alternatives on three equally weighted benefit criteria. The
    #: second and third are excellent twice over and nearly disqualified once.
    FLAWED = np.array([[6.0, 6.0, 6.0], [9.0, 9.0, 0.6], [9.0, 9.0, 0.05]])
    EQUAL = np.array([1 / 3, 1 / 3, 1 / 3])
    BENEFIT = ["benefit", "benefit", "benefit"]

    def test_a_near_zero_criterion_is_not_paid_for(self):
        scores = wpm(self.FLAWED, self.EQUAL, self.BENEFIT)
        assert scores[0] > scores[1] > scores[2], (
            "the balanced alternative must beat both spiky ones under a product"
        )

    def test_the_weighted_sum_reaches_the_opposite_conclusion(self):
        """Stated as a contrast, so the difference is the tested thing rather
        than an incidental property of this fixture."""
        criteria = [Criterion(f"c{i}", 1 / 3, "benefit") for i in range(3)]
        summed = rank(self.FLAWED, criteria, method="weighted_scoring").scores
        assert summed[1] > summed[0], (
            "the weighted sum should prefer the spiky alternative here"
        )
        product = wpm(self.FLAWED, self.EQUAL, self.BENEFIT)
        assert product[0] > product[1]

    def test_the_penalty_grows_as_the_weak_criterion_worsens(self):
        worsening = [
            wpm(np.array([[6.0, 6.0, 6.0], [9.0, 9.0, v]]), self.EQUAL, self.BENEFIT)[1]
            for v in (3.0, 1.0, 0.3, 0.05)
        ]
        assert worsening == sorted(worsening, reverse=True)


class TestDegenerateInput:
    def test_a_non_positive_value_is_refused_with_a_reason(self):
        matrix = MATRIX.copy()
        matrix[0, 0] = 0.0
        with pytest.raises(McdaError, match="strictly positive"):
            wpm(matrix, WEIGHTS, DIRECTIONS)

    def test_all_zero_weights_return_zeros(self):
        scores = wpm(MATRIX, np.zeros(4), DIRECTIONS)
        assert scores == pytest.approx(np.zeros(4))

    def test_a_flat_criterion_is_handled(self):
        matrix = MATRIX.copy()
        matrix[:, 3] = 5.0
        assert np.all(np.isfinite(wpm(matrix, WEIGHTS, DIRECTIONS)))


class TestThroughTheApi:
    def decision(self):
        criteria = [
            Criterion(name, float(w), d)
            for name, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]
        return Decision(MATRIX, criteria, LABELS)

    def test_it_is_registered(self):
        from mcdakit import method_names

        assert "wpm" in method_names()

    def test_rank_reproduces_the_reference_scores(self):
        result = rank(self.decision(), method="wpm")
        assert result.scores == pytest.approx(REFERENCE, abs=1e-8)

    def test_it_takes_the_matrix_as_measured(self):
        """Declared Wants.RAW: it resolves cost criteria itself, so the
        orientation step must not mirror them first. Getting this wrong
        inverts the ranking silently."""
        from mcdakit.methods import get as get_method

        assert get_method("wpm").wants.name == "RAW"

    def test_it_accepts_no_lambda(self):
        """Fixing lambda at zero is what distinguishes it from waspas, so a
        caller passing one has misunderstood and is told so by name rather
        than having it silently ignored."""
        with pytest.raises(McdaError, match=r"takes no keyword arguments.*lambda_"):
            rank(self.decision(), method="wpm", lambda_=0.5)

    def test_it_passes_the_conformance_suite(self):
        from mcdakit.methods import get as get_method
        from mcdakit.testing import check_method

        check_method(get_method("wpm"))
