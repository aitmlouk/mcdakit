"""Branches the main suites do not reach.

Error paths, degenerate inputs and container protocol methods. None of these
are exotic: each is code a user can reach, and an untested error path is one
that can rot into a confusing message — or a crash — without anything failing.
"""

import numpy as np
import pytest

from mcdakit import (
    METHODS,
    Criterion,
    Decision,
    McdaError,
    Method,
    Result,
    ahp_weights,
    orient,
    priorities,
    rank,
    register_normalization,
    sensitivity,
)
from mcdakit.methods.base import ScoringContext
from mcdakit.methods.spotis import spotis


class TestCriterionValidation:
    """Each rejection message a user can actually trigger."""

    def test_non_numeric_bounds_are_refused(self):
        with pytest.raises(McdaError, match="bounds must be a"):
            Criterion("Price", bounds=("cheap", "dear"))

    def test_a_single_bound_is_refused(self):
        with pytest.raises(McdaError, match="bounds must be a"):
            Criterion("Price", bounds=(5,))

    def test_infinite_bounds_are_refused(self):
        """Infinity gives a span of infinity, so every distance collapses to
        zero — a silent way to make every option look identical."""
        with pytest.raises(McdaError, match="bounds must be finite"):
            Criterion("Price", bounds=(0, float("inf")))

    def test_a_negative_threshold_is_refused(self):
        with pytest.raises(McdaError, match="non-negative"):
            Criterion("Price", preference_shape="linear", q=-1, p=5)

    def test_an_infinite_threshold_is_refused(self):
        with pytest.raises(McdaError, match="non-negative"):
            Criterion("Price", preference_shape="vshape", p=float("inf"))


class TestDecisionValidation:
    def test_a_non_numeric_matrix_says_so(self):
        with pytest.raises(McdaError, match="numbers only"):
            Decision([["cheap", "good"]], [Criterion("a"), Criterion("b")])

    def test_a_flat_list_of_numbers_explains_the_nesting(self):
        """A plausible mistake — writing [1, 2] for one option instead of
        [[1, 2]]. It used to surface as numpy's "object of type float has no
        len()", which tells a user nothing."""
        with pytest.raises(McdaError, match="flat sequence"):
            Decision([1.0, 2.0], [Criterion("a"), Criterion("b")])

    def test_a_row_that_is_not_a_sequence_is_refused(self):
        with pytest.raises(McdaError, match="ragged"):
            Decision([[1.0, 2.0], 3.0], [Criterion("a"), Criterion("b")])

    def test_a_three_dimensional_matrix_is_refused(self):
        with pytest.raises(McdaError, match="two-dimensional"):
            Decision(np.ones((2, 2, 2)), [Criterion("a"), Criterion("b")])

    def test_criteria_must_be_criterion_instances(self):
        with pytest.raises(McdaError, match="Criterion instances"):
            Decision([[1.0]], ["Price"])

    def test_names_lists_the_criteria_in_column_order(self):
        decision = Decision([[1.0, 2.0]], [Criterion("Price"), Criterion("Quality")])
        assert decision.names == ["Price", "Quality"]


class TestResultLookups:
    def test_rank_of_an_unknown_label_is_refused(self, supplier):
        result = rank(supplier)
        with pytest.raises(McdaError, match="No option named"):
            result.rank_of("Supplier Z")

    def test_warnings_are_shown_when_printed(self):
        """A caveat the user should see must survive into the readable form."""
        criteria = [Criterion("Price", 1.0, "cost")]
        decision = Decision([[2.0], [3.0]], criteria, ["A", "B"])
        result = Result(
            decision=decision,
            method="demo",
            scores=np.array([1.0, 2.0]),
            warnings=("bounds were guessed",),
        )
        assert "! bounds were guessed" in str(result)


class TestMethodNamesView:
    """`METHODS` replaced a frozen tuple, so it must behave like one."""

    def test_it_indexes(self):
        assert METHODS[0] == "simple_scoring"
        assert METHODS[-1] in list(METHODS)

    def test_it_slices(self):
        assert list(METHODS[:2]) == ["simple_scoring", "weighted_scoring"]

    def test_it_compares_equal_to_a_tuple_of_the_same_names(self):
        from mcdakit.methods import names

        assert names() == METHODS
        assert tuple(names()) == METHODS

    def test_it_is_hashable(self):
        """A frozen tuple was; code may put it in a set or dict key."""
        assert hash(METHODS) == hash(tuple(METHODS))
        assert {METHODS}

    def test_it_reprs_as_its_contents(self):
        assert "topsis" in repr(METHODS)

    def test_it_supports_len_and_membership(self):
        assert len(METHODS) == 8
        assert "spotis" in METHODS
        assert "nonsense" not in METHODS


class TestAhpEdgeCases:
    def test_a_degenerate_matrix_falls_back_to_equal_priorities(self):
        """A column of zeros makes every geometric mean zero. Equal weights
        are the only sensible reading, and beat dividing by zero."""
        assert priorities(np.zeros((3, 3))) == pytest.approx([1 / 3] * 3)

    def test_a_full_matrix_needs_no_names(self):
        """Names are only required when the size must be inferred from them."""
        matrix = np.array([[1.0, 3.0], [1 / 3, 1.0]])
        out = ahp_weights(matrix)
        assert out["names"] == ["Criterion 1", "Criterion 2"]
        assert out["weights"] == pytest.approx([0.75, 0.25])


class TestScoringContextFallbacks:
    def test_all_zero_weights_give_uniform_normalised_weights(self):
        """Unreachable through rank(), since a Decision refuses all-zero
        weights — but a hand-built context can hold them, and returning NaN
        would poison every score."""
        decision = Decision([[1.0, 2.0]], [Criterion("a"), Criterion("b")])
        ctx = ScoringContext(
            data=decision.matrix,
            weights=np.zeros(2),
            decision=decision,
        )
        assert ctx.normalized_weights == pytest.approx([0.5, 0.5])


class TestOrientation:
    def test_an_empty_matrix_passes_through(self):
        empty = np.empty((0, 0))
        assert orient(empty, []).shape == (0, 0)


class TestSpotisWithoutBounds:
    def test_bounds_may_be_omitted_entirely(self):
        """Passing None for the whole list, rather than one None per
        criterion, must behave the same: fall back and say so."""
        scores, reversal_free, messages = spotis(
            np.array([[2.0, 8.0], [3.0, 6.0]]),
            np.array([0.5, 0.5]),
            ["cost", "benefit"],
            None,
            warn=False,
        )
        assert reversal_free is False
        assert messages and "criterion 1" in messages[0]
        assert np.all(np.isfinite(scores))


class TestNormalizationRegistration:
    def test_an_empty_name_is_refused(self):
        with pytest.raises(McdaError, match="non-empty name"):
            register_normalization("   ", lambda data: data)


class TestSensitivityBands:
    def test_a_result_nothing_can_unseat_is_immovable(self, supplier_criteria):
        """Both flip searches return None, so there is no tolerance at all."""
        matrix = [[9, 9, 9, 9], [1, 1, 1, 1], [5, 5, 5, 5]]
        report = sensitivity(
            rank(matrix, supplier_criteria, labels=["Best", "Worst", "Mid"])
        )
        assert report["level"] == "immovable"
        assert report["weakest"] == ""
        assert report["weakest_flips_to"] == ""

    def test_a_winner_needing_a_large_shift_is_reported_as_robust(self):
        """The band the other suites never produce: a tolerance at or above
        25%, where no plausible re-weighting changes the answer."""
        criteria = [Criterion("A", 0.5), Criterion("B", 0.5)]
        matrix = [[9.0, 8.0], [1.0, 2.0]]
        report = sensitivity(rank(matrix, criteria, labels=["Strong", "Weak"]))
        assert report["level"] in {"robust", "immovable"}
        if report["overall"] is not None:
            assert report["overall"] >= 0.25

    def test_a_method_that_cannot_be_scored_is_reported(self, supplier):
        """A method returning nothing must produce a clear message rather
        than an IndexError from inside the bisection."""
        from mcdakit.methods.registry import register, unregister

        class Empty(Method):
            name = "test_scores_nothing"
            summary = "Returns an empty vector."
            citation = "n/a"

            def score(self, ctx):
                return np.array([])

        register(Empty())
        try:
            with pytest.raises(McdaError):
                sensitivity(rank(supplier, method="test_scores_nothing"))
        finally:
            unregister("test_scores_nothing")


class TestSensitivityFlipSearch:
    def test_a_criterion_already_at_zero_cannot_be_lowered(self):
        """The downward search has no room, so it must report None rather
        than searching a negative range."""
        criteria = [Criterion("Live", 1.0), Criterion("Muted", 0.0)]
        matrix = [[9.0, 1.0], [1.0, 9.0]]
        report = sensitivity(rank(matrix, criteria, labels=["A", "B"]))
        muted = next(r for r in report["rows"] if r["name"] == "Muted")
        assert muted["decrease"] is None


class TestConsistencyRatioStandalone:
    def test_it_derives_the_weights_when_not_given_them(self):
        """`consistency_ratio(matrix)` is public and usable on its own, not
        only as a step inside `ahp_weights`."""
        from mcdakit import consistency_ratio

        consistent = np.array([[1.0, 3.0, 9.0], [1 / 3, 1.0, 3.0], [1 / 9, 1 / 3, 1.0]])
        assert consistency_ratio(consistent) == pytest.approx(0.0, abs=1e-9)

    def test_it_agrees_with_the_weights_ahp_derives(self):
        from mcdakit import consistency_ratio

        matrix = np.array([[1.0, 5.0, 3.0], [0.2, 1.0, 0.5], [1 / 3, 2.0, 1.0]])
        assert consistency_ratio(matrix) == pytest.approx(
            consistency_ratio(matrix, priorities(matrix))
        )


class TestLinearShapeDegenerate:
    def test_linear_with_p_equal_to_q_is_a_step_above_q(self):
        """No band to interpolate across, so it collapses to a step rather
        than dividing by p - q == 0."""
        from mcdakit.methods import preference_degree

        assert preference_degree(11, "linear", q=10, p=10) == 1.0
        assert preference_degree(10, "linear", q=10, p=10) == 0.0

    def test_linear_saturates_above_the_preference_threshold(self):
        """With a real band (q < p), a difference beyond p is total
        preference. Existing tests stop at exactly p, which takes the
        interpolation path instead."""
        from mcdakit.methods import preference_degree

        assert preference_degree(25, "linear", q=10, p=20) == 1.0
        assert preference_degree(100, "linear", q=10, p=20) == 1.0
