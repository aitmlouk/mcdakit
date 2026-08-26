"""SPOTIS: the reversal-freedom claim, and the honesty of its fallback."""

import warnings

import numpy as np
import pytest

from mcdakit import (
    METHODS,
    BoundsWarning,
    Criterion,
    Decision,
    rank,
    reversal_check,
)


class TestReversalFreedom:
    def test_removing_a_loser_cannot_reorder_the_survivors(self, procurement):
        """The whole claim, on the worked procurement case."""
        check = reversal_check(procurement, method="spotis")
        assert check["reversed"] is False, check["cases"]

    def test_scores_of_the_survivors_do_not_move_at_all(self, procurement):
        """Stronger than order preservation: the numbers themselves are
        untouched, because each option is measured against the fixed ideal
        point and never against the others."""
        full = rank(procurement, method="spotis")
        reduced = rank(procurement.without(3), method="spotis")
        for label in reduced.decision.labels:
            assert reduced.score_of(label) == pytest.approx(
                full.score_of(label), abs=1e-12)

    def test_the_documented_supplier_case(self, procurement):
        """Kestrel beats Nordpack with the rejected option present, and still
        beats it once that option is removed. Under weighted scoring it does
        not — which is why this case is in the README."""
        assert rank(procurement, method="spotis").winner == "Kestrel Supply"
        survivors = procurement.without(3)
        assert rank(survivors, method="spotis").winner == "Kestrel Supply"

        before = rank(procurement, method="weighted_scoring")
        after = rank(survivors, method="weighted_scoring")
        assert before.winner == "Kestrel Supply"
        assert after.winner == "Nordpack", (
            "weighted scoring must show the reversal the README describes")

    def test_adding_an_option_cannot_reorder_the_existing_ones(self, procurement):
        """Reversal-freedom runs both ways: a new candidate joining the
        shortlist must not reshuffle the incumbents."""
        before = rank(procurement, method="spotis")
        extended = Decision(
            np.vstack([procurement.matrix, [[3.9, 2.0, 33, 2.0]]]),
            procurement.criteria,
            [*procurement.labels, "Latecomer"],
        )
        after = rank(extended, method="spotis")
        kept = [label for label in after.order if label != "Latecomer"]
        assert kept == before.order

    def test_a_reversal_prone_method_is_caught_by_the_same_check(
        self, procurement
    ):
        """The check itself must be capable of reporting True, or the SPOTIS
        result above would prove nothing."""
        check = reversal_check(procurement, method="weighted_scoring")
        assert check["reversed"] is True
        assert check["cases"][0]["removed"] == "Bytharm"


class TestBounds:
    def test_the_ideal_point_comes_from_bounds_not_from_the_data(self):
        """Two problems sharing bounds but not options must score their
        shared option identically."""
        criteria = [
            Criterion("Price", 0.5, "cost", bounds=(0, 100)),
            Criterion("Quality", 0.5, "benefit", bounds=(0, 10)),
        ]
        shared = [50.0, 5.0]
        one = rank([shared, [10.0, 1.0]], criteria, "spotis", ["X", "Cheap"])
        two = rank([shared, [90.0, 9.0]], criteria, "spotis", ["X", "Dear"])
        assert one.score_of("X") == pytest.approx(two.score_of("X"))

    def test_an_option_on_the_ideal_point_scores_zero(self):
        """Scores are negated distances, so zero is perfection."""
        criteria = [
            Criterion("Price", 0.5, "cost", bounds=(2.0, 4.0)),
            Criterion("Quality", 0.5, "benefit", bounds=(0, 10)),
        ]
        result = rank([[2.0, 10.0], [4.0, 0.0]], criteria, "spotis",
                      ["Perfect", "Awful"])
        assert result.score_of("Perfect") == pytest.approx(0.0)
        assert result.score_of("Awful") == pytest.approx(-1.0)

    def test_missing_bounds_warn_and_withdraw_the_guarantee(self):
        """The worst possible failure for this package would be silently
        returning a number that looks reversal-free and is not."""
        criteria = [
            Criterion("Price", 0.5, "cost"),
            Criterion("Quality", 0.5, "benefit", bounds=(0, 10)),
        ]
        with pytest.warns(BoundsWarning, match="NOT rank-reversal-free"):
            result = rank([[2.0, 8.0], [3.0, 6.0]], criteria, "spotis")

        assert result.reversal_free is False
        assert result.warnings, "the shortfall must be readable from the result"
        assert "criterion 1" in result.warnings[0]

    def test_full_bounds_carry_the_guarantee(self, procurement):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rank(procurement, method="spotis")
        assert result.reversal_free is True
        assert result.warnings == ()

    def test_the_warning_can_be_promoted_to_an_error(self):
        """A pipeline that depends on the guarantee must be able to insist."""
        criteria = [Criterion("Price", 1.0, "cost")]
        with warnings.catch_warnings():
            warnings.simplefilter("error", BoundsWarning)
            with pytest.raises(BoundsWarning):
                rank([[2.0], [3.0]], criteria, "spotis")

    def test_only_spotis_claims_reversal_freedom(self, procurement):
        """No other method may set the flag, however well it happens to do."""
        for method in METHODS:
            if method == "spotis":
                continue
            assert rank(procurement, method=method).reversal_free is False

    def test_a_degenerate_bound_does_not_divide_by_zero(self):
        """bounds validation forbids lo == hi, but a criterion can still be
        constant across the options; the distance must stay finite."""
        criteria = [
            Criterion("Flat", 0.5, "benefit", bounds=(0, 10)),
            Criterion("Real", 0.5, "benefit", bounds=(0, 10)),
        ]
        scores = rank([[5.0, 1.0], [5.0, 9.0]], criteria, "spotis").scores
        assert np.all(np.isfinite(scores))


class TestPaperExample:
    """The car-selection problem from Dezert et al., FUSION 2020, Section 4.

    Four cars, four criteria, with the bounds the paper sets. Expected values
    are cross-checked against `pymcdm` 1.4.0's SPOTIS, the reference
    implementation, rather than transcribed from the paper's table — a number
    verified against runnable code is worth more than one copied by hand.
    """

    CRITERIA = [
        Criterion("C1", 0.2, "benefit", bounds=(-5, 12)),
        Criterion("C2", 0.3, "cost", bounds=(-6, 10)),
        Criterion("C3", 0.4, "benefit", bounds=(-8, 5)),
        Criterion("C4", 0.1, "benefit", bounds=(-4, 6)),
    ]
    MATRIX = [
        [10.5, -3.1, 1.7, 3.4],
        [-4.7, 0.0, 3.4, 5.6],
        [8.1, 0.3, 1.3, 2.1],
        [3.2, 7.3, -5.3, 1.9],
    ]
    LABELS = ["A1", "A2", "A3", "A4"]

    # SPOTIS distances, smallest is best. mcdakit negates these.
    REFERENCE = {
        "A1": 0.19956052,
        "A2": 0.36220136,
        "A3": 0.31685351,
        "A4": 0.71082749,
    }

    def test_matches_the_reference_implementation(self):
        result = rank(self.MATRIX, self.CRITERIA, "spotis", labels=self.LABELS)
        for label, distance in self.REFERENCE.items():
            assert result.score_of(label) == pytest.approx(-distance, abs=1e-8)

    def test_gives_the_papers_ranking(self):
        result = rank(self.MATRIX, self.CRITERIA, "spotis", labels=self.LABELS)
        assert result.order == ["A1", "A3", "A2", "A4"]

    def test_this_problem_is_reversal_free_too(self):
        check = reversal_check(self.MATRIX, self.CRITERIA, "spotis",
                               labels=self.LABELS)
        assert check["reversed"] is False
