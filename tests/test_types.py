"""Inputs are validated where the mistake was made, not deep inside numpy."""

import numpy as np
import pytest

from mcdakit import Criterion, Decision, McdaError, rank


class TestCriterion:
    def test_defaults_to_a_benefit_criterion(self):
        assert Criterion("Quality").direction == "benefit"
        assert not Criterion("Quality").is_cost

    @pytest.mark.parametrize("bad", ["Benefit", "maximise", "", None])
    def test_unknown_direction_is_refused(self, bad):
        with pytest.raises(McdaError, match="direction"):
            Criterion("X", direction=bad)

    def test_negative_weight_is_refused(self):
        with pytest.raises(McdaError, match="negative"):
            Criterion("X", weight=-1)

    def test_zero_weight_is_allowed_on_one_criterion(self):
        """Muting one criterion is a legitimate what-if; muting all is not."""
        assert Criterion("X", weight=0).weight == 0.0

    def test_nan_weight_is_refused(self):
        with pytest.raises(McdaError, match="finite"):
            Criterion("X", weight=float("nan"))

    def test_bounds_must_increase(self):
        with pytest.raises(McdaError, match="lo < hi"):
            Criterion("X", bounds=(10, 2))
        with pytest.raises(McdaError, match="lo < hi"):
            Criterion("X", bounds=(5, 5))

    def test_bounds_must_be_a_pair(self):
        with pytest.raises(McdaError, match="pair"):
            Criterion("X", bounds=(1, 2, 3))

    def test_unnamed_criteria_are_refused(self):
        with pytest.raises(McdaError, match="name"):
            Criterion("   ")

    def test_preference_thresholds_must_be_ordered(self):
        with pytest.raises(McdaError, match="must exceed"):
            Criterion("X", preference_shape="linear", q=50, p=20)

    def test_unknown_preference_shape_is_refused(self):
        with pytest.raises(McdaError, match="preference_shape"):
            Criterion("X", preference_shape="sigmoid")


class TestDecision:
    def test_a_ragged_matrix_says_it_is_ragged(self):
        """Regression: this used to surface as numpy's 'setting an array
        element with a sequence', which tells a user nothing."""
        with pytest.raises(McdaError, match="ragged"):
            Decision([[1, 2], [3]], [Criterion("A"), Criterion("B")])

    def test_column_count_must_match_the_criteria(self):
        with pytest.raises(McdaError, match="one column per criterion"):
            Decision([[1, 2, 3]], [Criterion("A"), Criterion("B")])

    def test_label_count_must_match_the_options(self):
        with pytest.raises(McdaError, match="one label per row"):
            Decision([[1], [2]], [Criterion("A")], ["only one"])

    def test_labels_default_to_numbered_options(self):
        assert Decision([[1], [2]], [Criterion("A")]).labels == ("Option 1", "Option 2")

    def test_all_zero_weights_are_refused(self):
        """Every criterion muted is not a decision problem, and silently
        returning zeros would look like an answer."""
        with pytest.raises(McdaError, match="all zero"):
            Decision([[1, 2]], [Criterion("A", 0), Criterion("B", 0)])

    def test_missing_values_are_refused(self):
        with pytest.raises(McdaError, match="NaN"):
            Decision([[1.0, np.nan]], [Criterion("A"), Criterion("B")])

    def test_empty_problems_are_refused(self):
        with pytest.raises(McdaError, match="at least one option"):
            Decision(np.empty((0, 1)), [Criterion("A")])
        with pytest.raises(McdaError, match="at least one criterion"):
            Decision([[1]], [])

    def test_weights_are_normalised_on_demand_only(self, supplier):
        assert supplier.weights.sum() == pytest.approx(1.0)
        assert supplier.normalized_weights.sum() == pytest.approx(1.0)

    def test_weight_scale_does_not_change_the_ranking(self, supplier_criteria):
        """40/30/20/10 and 0.4/0.3/0.2/0.1 are the same input."""
        big = [
            Criterion(c.name, c.weight * 100, c.direction) for c in supplier_criteria
        ]
        matrix = [[9, 5, 7, 6], [6, 9, 8, 9], [7, 7, 6, 5]]
        assert rank(matrix, big).order == rank(matrix, supplier_criteria).order

    def test_without_drops_exactly_one_option(self, supplier):
        reduced = supplier.without(1)
        assert reduced.labels == ("Supplier A", "Supplier C")
        assert reduced.n_options == 2
        assert supplier.n_options == 3, "the original must not be mutated"

    def test_without_rejects_an_index_that_is_not_there(self, supplier):
        with pytest.raises(McdaError, match="No option at index"):
            supplier.without(9)


class TestResult:
    def test_ties_share_a_rank_and_skip_the_next(self, supplier_criteria):
        """Competition ranking: two tied firsts are 1, 1, 3 — never 1, 1, 2."""
        flat = [[5, 5, 5, 5]] * 3
        result = rank(flat, supplier_criteria, labels=["A", "B", "C"])
        assert list(result.ranks) == [1, 1, 1]

    def test_ranks_follow_competition_convention(self, supplier_criteria):
        matrix = [[9, 9, 9, 9], [9, 9, 9, 9], [1, 1, 1, 1], [0, 0, 0, 0]]
        result = rank(matrix, supplier_criteria, labels=["A", "B", "C", "D"])
        assert list(result.ranks) == [1, 1, 3, 4]

    def test_lookup_by_label(self, supplier):
        result = rank(supplier)
        assert result.rank_of("Supplier B") == 1
        assert result.score_of("Supplier B") == pytest.approx(0.6, abs=1e-4)
        with pytest.raises(McdaError, match="No option named"):
            result.score_of("Supplier Z")

    def test_str_is_readable(self, supplier):
        text = str(rank(supplier))
        assert "1. Supplier B" in text
        assert "weighted_scoring" in text
