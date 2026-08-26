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
