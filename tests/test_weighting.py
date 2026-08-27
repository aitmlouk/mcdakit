"""Deriving weights from the data instead of eliciting them.

The premise every scheme here shares: a criterion on which all the options
score alike cannot separate them, so it carries no information. These tests
hold each scheme to that, and to the definition it claims to implement.
"""

import numpy as np
import pytest

from mcdakit import (
    Criterion,
    McdaError,
    available_weightings,
    critic_weights,
    derive_weights,
    entropy_weights,
    equal_weights,
    rank,
    register_weighting,
    std_weights,
)
from mcdakit.weighting import WEIGHTINGS, get_weighting

#: Column 0 is identical for every option and so carries no information;
#: column 1 separates them.
FLAT_AND_INFORMATIVE = np.array([[5.0, 1.0], [5.0, 9.0], [5.0, 5.0]])

INFORMATIVE = np.array([[1.0, 1.0], [9.0, 2.0], [5.0, 1.5]])


@pytest.fixture
def restore_schemes():
    saved = dict(WEIGHTINGS)
    yield
    WEIGHTINGS.clear()
    WEIGHTINGS.update(saved)


class TestTheSharedPremise:
    """The one property every informative scheme must have."""

    @pytest.mark.parametrize("scheme", ["entropy", "std", "critic"])
    def test_a_criterion_nobody_differs_on_gets_no_weight(self, scheme):
        weights = derive_weights(FLAT_AND_INFORMATIVE, scheme)
        assert weights[0] == pytest.approx(0.0, abs=1e-9)
        assert weights[1] == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("scheme", ["entropy", "std", "critic"])
    def test_it_holds_when_the_flat_value_is_negative(self, scheme):
        """Regression: shifting by the column minimum turned a constant column
        into all zeros, which read as zero entropy and therefore *maximum*
        weight — the exact inverse of what entropy weighting means."""
        negative = np.array([[-5.0, 1.0], [-5.0, 9.0], [-5.0, 5.0]])
        assert derive_weights(negative, scheme)[0] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("scheme", ["equal", "entropy", "std", "critic"])
    def test_weights_sum_to_one_and_are_non_negative(self, scheme):
        weights = derive_weights(INFORMATIVE, scheme)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0)

    @pytest.mark.parametrize("scheme", ["equal", "entropy", "std", "critic"])
    def test_one_weight_per_criterion(self, scheme):
        matrix = np.arange(12, dtype=float).reshape(3, 4)
        assert derive_weights(matrix, scheme).shape == (4,)


class TestEqual:
    def test_it_is_uniform(self):
        assert equal_weights(INFORMATIVE) == pytest.approx([0.5, 0.5])

    def test_it_ignores_the_data_entirely(self):
        """The baseline: if an objective scheme cannot beat this on your
        problem, it is adding complexity rather than information."""
        assert equal_weights(INFORMATIVE) == pytest.approx(
            equal_weights(FLAT_AND_INFORMATIVE)
        )


class TestEntropy:
    def test_a_perfectly_flat_matrix_falls_back_to_equal(self):
        """No criterion separates anything, so none can be more important."""
        flat = np.full((3, 3), 7.0)
        assert derive_weights(flat, "entropy") == pytest.approx([1 / 3] * 3)

    def test_a_single_option_carries_no_information(self):
        """With one option there is nothing to separate."""
        assert entropy_weights(np.array([[1.0, 5.0]])) == pytest.approx([0.5, 0.5])

    def test_a_sharper_split_earns_more_weight(self):
        """A criterion that separates the options decisively should outweigh
        one that barely distinguishes them."""
        matrix = np.array([[1.0, 5.0], [9.0, 5.1], [5.0, 5.05]])
        weights = entropy_weights(matrix)
        assert weights[0] > weights[1]

    def test_it_is_scale_invariant(self):
        """Entropy works on proportions, so multiplying a column by a constant
        must not change its weight — this is what distinguishes it from
        std_weights."""
        scaled = INFORMATIVE.copy()
        scaled[:, 0] *= 1000.0
        assert entropy_weights(INFORMATIVE) == pytest.approx(
            entropy_weights(scaled), abs=1e-9
        )


class TestStd:
    def test_it_weights_by_spread(self):
        matrix = np.array([[0.0, 4.0], [10.0, 6.0]])
        # Population standard deviations are 5.0 and 1.0.
        weights = std_weights(matrix)
        assert weights[0] == pytest.approx(5 / 6)
        assert weights[1] == pytest.approx(1 / 6)

    def test_it_is_sensitive_to_units(self):
        """Documented weakness: rescaling a column changes its weight, which
        is why it suits already-comparable scales like ratings."""
        scaled = INFORMATIVE.copy()
        scaled[:, 0] *= 1000.0
        assert not np.allclose(std_weights(INFORMATIVE), std_weights(scaled))


class TestCritic:
    def test_duplicate_criteria_share_the_weight(self):
        """The point of CRITIC over std: two criteria measuring the same thing
        should not each count in full, or that dimension is counted twice."""
        varying = np.array([1.0, 9.0, 5.0])
        duplicated = np.column_stack([varying, varying, np.array([1.0, 2.0, 9.0])])
        weights = critic_weights(duplicated)
        assert weights[0] == pytest.approx(weights[1]), (
            "identical criteria must be weighted identically"
        )
        assert weights[2] > weights[0], (
            "the independent criterion should outweigh either duplicate"
        )

    def test_a_single_criterion_takes_all_the_weight(self):
        assert critic_weights(np.array([[1.0], [9.0]])) == pytest.approx([1.0])

    def test_a_single_option_falls_back_to_equal(self):
        assert critic_weights(np.array([[1.0, 5.0]])) == pytest.approx([0.5, 0.5])

    def test_a_constant_column_does_not_poison_the_correlations(self):
        """Its correlation is undefined; treating it as uncorrelated is the
        neutral choice and must not produce NaN weights elsewhere."""
        matrix = np.array([[5.0, 1.0, 2.0], [5.0, 9.0, 8.0], [5.0, 5.0, 3.0]])
        weights = critic_weights(matrix)
        assert np.all(np.isfinite(weights))
        assert weights[0] == pytest.approx(0.0, abs=1e-9)


class TestDirections:
    def test_supplying_directions_orients_the_matrix(self):
        """It matters for CRITIC: a cost and a benefit criterion that move
        together look opposed until one is flipped."""
        # Asymmetric on purpose: with two perfectly anti-correlated columns
        # the weights are symmetric either way, so the test would pass
        # vacuously.
        matrix = np.array([[1.0, 9.0, 2.0], [5.0, 4.0, 8.0], [9.0, 1.0, 3.0]])
        naive = critic_weights(matrix)
        oriented = critic_weights(matrix, ["benefit", "cost", "benefit"])
        assert not np.allclose(naive, oriented)

    def test_a_direction_count_mismatch_is_refused(self):
        with pytest.raises(McdaError, match="directions for"):
            entropy_weights(INFORMATIVE, ["benefit"])


class TestRegistration:
    def test_a_user_scheme_works_without_touching_the_library(self, restore_schemes):
        @register_weighting("test_range", summary="Weight by observed range.")
        def _by_range(matrix, directions=None):
            data = np.asarray(matrix, dtype=float)
            return data.max(axis=0) - data.min(axis=0)

        assert "test_range" in available_weightings()
        weights = derive_weights(FLAT_AND_INFORMATIVE, "test_range")
        assert weights == pytest.approx([0.0, 1.0])

    def test_a_bare_callable_needs_no_registration(self):
        weights = derive_weights(
            INFORMATIVE, lambda matrix, directions=None: np.array([3.0, 1.0])
        )
        assert weights == pytest.approx([0.75, 0.25])

    def test_a_duplicate_name_is_refused(self, restore_schemes):
        with pytest.raises(McdaError, match="already registered"):
            register_weighting("entropy", entropy_weights)

    def test_a_duplicate_can_be_replaced_deliberately(self, restore_schemes):
        register_weighting("entropy", entropy_weights, replace=True)
        assert "entropy" in available_weightings()

    def test_an_unknown_name_lists_the_real_ones(self):
        with pytest.raises(McdaError, match="Unknown weighting"):
            get_weighting("nonsense")

    def test_a_non_callable_is_refused(self, restore_schemes):
        with pytest.raises(McdaError, match="must be callable"):
            register_weighting("test_bad", "not a function")

    def test_an_empty_name_is_refused(self, restore_schemes):
        with pytest.raises(McdaError, match="non-empty name"):
            register_weighting("  ", equal_weights)


class TestOutputIsValidated:
    def test_a_wrong_weight_count_is_caught(self):
        with pytest.raises(McdaError, match="weights for 2 criteria"):
            derive_weights(INFORMATIVE, lambda m, d=None: np.array([1.0]))

    def test_a_negative_weight_is_refused(self):
        """A criterion cannot count against itself."""
        with pytest.raises(McdaError, match="negative weight"):
            derive_weights(INFORMATIVE, lambda m, d=None: np.array([1.0, -1.0]))

    def test_a_non_finite_weight_is_refused(self):
        with pytest.raises(McdaError, match="non-finite"):
            derive_weights(INFORMATIVE, lambda m, d=None: np.array([1.0, np.nan]))

    def test_all_zero_weights_fall_back_to_equal(self):
        """No criterion carried information; equal weights are the only
        honest reading, and beat dividing by zero."""
        weights = derive_weights(INFORMATIVE, lambda m, d=None: np.zeros(2))
        assert weights == pytest.approx([0.5, 0.5])

    def test_a_non_callable_scheme_is_refused(self):
        with pytest.raises(McdaError, match="name or a callable"):
            derive_weights(INFORMATIVE, 42)


class TestInputIsValidated:
    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(McdaError, match="two-dimensional"):
            entropy_weights(np.array([1.0, 2.0, 3.0]))

    def test_an_empty_matrix_is_refused(self):
        with pytest.raises(McdaError, match="at least one option"):
            entropy_weights(np.empty((0, 2)))

    def test_a_matrix_with_nan_is_refused(self):
        with pytest.raises(McdaError, match="NaN"):
            entropy_weights(np.array([[1.0, np.nan]]))


class TestFeedingRank:
    def test_derived_weights_can_drive_a_ranking(self):
        """A weighting nobody can act on is decoration."""
        matrix = [[2.0, 8.0], [3.0, 6.0], [4.0, 4.0]]
        directions = ["cost", "benefit"]
        weights = derive_weights(matrix, "entropy", directions)

        criteria = [
            Criterion(name, float(weight), direction)
            for name, weight, direction in zip(
                ("Price", "Quality"), weights, directions
            )
        ]
        result = rank(matrix, criteria, labels=["A", "B", "C"])
        assert result.winner == "A"

    def test_a_flat_criterion_is_muted_rather_than_breaking_the_rank(self):
        """A zero weight is legal on one criterion; Decision only refuses
        weights that are all zero."""
        matrix = [[5.0, 1.0], [5.0, 9.0]]
        weights = derive_weights(matrix, "entropy")
        criteria = [
            Criterion("Flat", float(weights[0])),
            Criterion("Real", float(weights[1])),
        ]
        assert rank(matrix, criteria, labels=["A", "B"]).winner == "B"
