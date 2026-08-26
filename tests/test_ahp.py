"""AHP: pairwise judgements to weights, with consistency reported."""

import numpy as np
import pytest

from mcdakit import (
    CONSISTENCY_LIMIT,
    RANDOM_INDEX,
    Criterion,
    McdaError,
    ahp_weights,
    comparison_matrix,
    consistency_ratio,
    priorities,
    rank,
)


class TestPriorities:
    def test_consistent_judgements_give_the_expected_weights(self):
        """Price 3x Quality and Quality 3x Delivery implies Price 9x Delivery.
        A perfectly consistent matrix has ratio zero and priorities in 9:3:1."""
        out = ahp_weights(
            {(0, 1): 3, (0, 2): 9, (1, 2): 3},
            names=["Price", "Quality", "Delivery"],
        )
        assert out["consistency_ratio"] == pytest.approx(0.0, abs=1e-9)
        assert out["consistent"]
        assert out["weights"][0] == pytest.approx(9 / 13, abs=1e-4)
        assert out["weights"][1] == pytest.approx(3 / 13, abs=1e-4)
        assert out["weights"][2] == pytest.approx(1 / 13, abs=1e-4)
        assert out["weights"].sum() == pytest.approx(1.0)

    def test_two_criteria_are_always_consistent(self):
        """With one judgement there is nothing to contradict."""
        out = ahp_weights({(0, 1): 3.0}, names=["One", "Two"])
        assert out["consistency_ratio"] == pytest.approx(0.0)
        assert out["weights"] == pytest.approx([0.75, 0.25])

    def test_equal_judgements_give_equal_weights(self):
        out = ahp_weights({}, names=["A", "B", "C", "D"])
        assert out["weights"] == pytest.approx([0.25] * 4)

    def test_priorities_always_sum_to_one(self, rng):
        for _ in range(50):
            size = int(rng.integers(2, 8))
            upper = {
                (i, j): float(rng.choice([1 / 9, 1 / 5, 1 / 3, 1, 3, 5, 9]))
                for i in range(size)
                for j in range(i + 1, size)
            }
            weights = priorities(comparison_matrix(size, upper))
            assert weights.sum() == pytest.approx(1.0)
            assert np.all(weights > 0)


class TestConsistency:
    def test_circular_judgements_are_flagged(self):
        """A over B, B over C, but C over A cannot all hold."""
        out = ahp_weights(
            {(0, 1): 5, (1, 2): 5, (0, 2): 0.2}, names=["X", "Y", "Z"])
        assert out["consistency_ratio"] > CONSISTENCY_LIMIT
        assert not out["consistent"]

    def test_inconsistency_is_reported_not_raised(self):
        """The analyst decides whether to proceed; taking that away would be
        presumptuous."""
        out = ahp_weights(
            {(0, 1): 9, (1, 2): 9, (0, 2): 1 / 9}, names=["X", "Y", "Z"])
        assert not out["consistent"]
        assert out["weights"].sum() == pytest.approx(1.0), (
            "weights must still be usable")

    def test_the_random_index_covers_the_usable_range(self):
        """Saaty's table is what makes the ratio comparable across sizes."""
        for size in range(3, 11):
            assert size in RANDOM_INDEX
            assert RANDOM_INDEX[size] > 0

    def test_small_matrices_are_trivially_consistent(self):
        assert consistency_ratio(np.array([[1.0, 3.0], [1 / 3, 1.0]])) == 0.0


class TestComparisonMatrix:
    def test_it_is_reciprocal(self):
        matrix = comparison_matrix(3, {(0, 1): 3, (1, 2): 5})
        assert matrix[1, 0] == pytest.approx(1 / 3)
        assert matrix[2, 1] == pytest.approx(1 / 5)
        assert np.allclose(np.diag(matrix), 1.0)

    def test_unjudged_pairs_default_to_equal(self):
        assert comparison_matrix(3, {(0, 1): 3})[0, 2] == 1.0

    def test_a_criterion_cannot_be_compared_with_itself(self):
        with pytest.raises(McdaError, match="with itself"):
            comparison_matrix(2, {(0, 0): 3})

    def test_out_of_range_pairs_are_refused(self):
        with pytest.raises(McdaError, match="out of range"):
            comparison_matrix(2, {(0, 5): 3})

    def test_a_non_positive_ratio_is_refused(self):
        with pytest.raises(McdaError, match="positive ratio"):
            comparison_matrix(2, {(0, 1): 0})


class TestRefusals:
    def test_a_single_criterion_is_refused(self):
        with pytest.raises(McdaError, match="at least two criteria"):
            ahp_weights(np.array([[1.0]]))

    def test_pairs_without_names_are_refused(self):
        with pytest.raises(McdaError, match="names are required"):
            ahp_weights({(0, 1): 3})

    def test_a_name_count_mismatch_is_refused(self):
        with pytest.raises(McdaError, match="names for a"):
            ahp_weights(np.ones((3, 3)), names=["only", "two"])

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(McdaError, match="square"):
            priorities(np.ones((2, 3)))


class TestFeedingRank:
    def test_the_weights_can_drive_a_ranking(self, supplier):
        """A weighting nobody can act on is decoration."""
        out = ahp_weights(
            {(0, 1): 1 / 9, (0, 2): 1 / 3, (0, 3): 1 / 3,
             (1, 2): 5, (1, 3): 5, (2, 3): 1},
            names=[c.name for c in supplier.criteria],
        )
        criteria = [
            Criterion(c.name, weight, c.direction)
            for c, weight in zip(supplier.criteria, out["weights"])
        ]
        result = rank(supplier.matrix, criteria, labels=supplier.labels)
        assert result.winner == "Supplier B", (
            "quality weighted far above price must favour the quality option")
