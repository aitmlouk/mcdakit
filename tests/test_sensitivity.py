"""Weight tolerance: how far someone can disagree before the answer changes."""

import numpy as np
import pytest

from mcdakit import (
    FRAGILE_THRESHOLD,
    METHODS,
    ROBUST_THRESHOLD,
    BoundsWarning,
    Criterion,
    McdaError,
    rank,
    sensitivity,
)
from mcdakit.sensitivity import _winner_index, _with_weights


class TestTolerance:
    def test_matches_the_hand_computation(self, supplier):
        """The fixture's winner flips when Price rises by 7.5% of the total
        weight, verified independently against the scoring formula."""
        report = sensitivity(rank(supplier, method="weighted_scoring"))
        assert report["winner"] == "Supplier B"

        price = next(r for r in report["rows"] if r["name"] == "Price")
        assert price["tolerance"] == pytest.approx(0.075, abs=1e-3)
        assert price["direction"] == "up"
        assert price["flips_to"] == "Supplier A"

    def test_the_reported_flip_point_actually_flips(self, supplier):
        """Nudge each weight past its stated tolerance and the winner must
        change; stop just short and it must not. Without this the numbers
        could drift and every other assertion here would still pass."""
        result = rank(supplier, method="weighted_scoring")
        report = sensitivity(result)
        weights = supplier.weights
        baseline = result.winner_index
        total = float(np.sum(weights))

        for index, row in enumerate(report["rows"]):
            if row["tolerance"] is None:
                continue
            sign = 1 if row["direction"] == "up" else -1
            over = weights.astype(float).copy()
            over[index] = max(
                0.0, over[index] + sign * row["tolerance"] * total * 1.02)
            assert _winner_index(supplier, over, "weighted_scoring") != baseline

            under = weights.astype(float).copy()
            under[index] = max(
                0.0, under[index] + sign * row["tolerance"] * total * 0.90)
            assert _winner_index(supplier, under, "weighted_scoring") == baseline

    def test_the_named_successor_is_the_one_that_takes_over(self, supplier):
        result = rank(supplier, method="weighted_scoring")
        report = sensitivity(result)
        total = float(np.sum(supplier.weights))

        for index, row in enumerate(report["rows"]):
            if row["tolerance"] is None:
                continue
            sign = 1 if row["direction"] == "up" else -1
            nudged = supplier.weights.astype(float).copy()
            nudged[index] = max(
                0.0, nudged[index] + sign * row["tolerance"] * total * 1.01)
            winner = _winner_index(supplier, nudged, "weighted_scoring")
            assert supplier.labels[winner] == row["flips_to"]


class TestBands:
    def test_a_dominant_option_is_immovable(self, supplier_criteria):
        """An option best on every criterion wins whatever the weights are, so
        there is no tolerance to report — not a tolerance of zero."""
        matrix = [[1, 1, 1, 1], [9, 9, 9, 9], [5, 5, 5, 5]]
        report = sensitivity(
            rank(matrix, supplier_criteria, labels=["A", "B", "C"]))
        assert report["level"] == "immovable"
        assert report["overall"] is None
        assert all(r["tolerance"] is None for r in report["rows"])

    def test_a_close_call_is_reported_as_fragile(self, supplier):
        report = sensitivity(rank(supplier, method="weighted_scoring"))
        assert report["level"] == "fragile"
        assert report["overall"] < FRAGILE_THRESHOLD

    def test_the_bands_follow_the_thresholds(self, supplier):
        """The label must be derivable from the number it claims to describe."""
        report = sensitivity(rank(supplier, method="weighted_scoring"))
        overall = report["overall"]
        if overall is None:
            expected = "immovable"
        elif overall < FRAGILE_THRESHOLD:
            expected = "fragile"
        elif overall < ROBUST_THRESHOLD:
            expected = "moderate"
        else:
            expected = "robust"
        assert report["level"] == expected

    def test_overall_is_the_smallest_row_tolerance(self, procurement):
        report = sensitivity(rank(procurement, method="spotis"))
        tolerances = [
            r["tolerance"] for r in report["rows"] if r["tolerance"] is not None
        ]
        assert report["overall"] == pytest.approx(min(tolerances))
        weakest = min(
            (r for r in report["rows"] if r["tolerance"] is not None),
            key=lambda r: r["tolerance"])
        assert report["weakest"] == weakest["name"]


class TestAcrossMethods:
    @pytest.mark.parametrize("method", METHODS)
    def test_it_runs_for_every_method(self, procurement, method):
        """Bisection works over whatever scoring function it is handed, so
        every method must be covered — including the step functions where an
        analytic shortcut would have been wrong."""
        report = sensitivity(rank(procurement, method=method))
        assert len(report["rows"]) == procurement.n_criteria
        assert report["level"] in {
            "fragile", "moderate", "robust", "immovable"}

    def test_simple_scoring_ignores_weights_so_nothing_flips(self, supplier):
        """It never reads the weights, so no weight change can move it. The
        honest answer is 'immovable', not a fabricated tolerance."""
        report = sensitivity(rank(supplier, method="simple_scoring"))
        assert report["level"] == "immovable"

    def test_spotis_does_not_re_warn_on_every_bisection_step(self):
        """A bisection re-scores hundreds of times. Repeating the missing-
        bounds warning each time would bury the one that matters."""
        criteria = [Criterion("Price", 0.5, "cost"), Criterion("Quality", 0.5)]
        with pytest.warns(BoundsWarning) as caught:
            result = rank([[2.0, 8.0], [3.0, 6.0]], criteria, "spotis")
            sensitivity(result)
        assert len(caught) == 1


class TestPurity:
    def test_it_does_not_disturb_the_result_it_was_given(self, supplier):
        result = rank(supplier, method="weighted_scoring")
        scores_before = result.scores.copy()
        weights_before = supplier.weights.copy()

        sensitivity(result)

        assert np.allclose(result.scores, scores_before)
        assert np.allclose(supplier.weights, weights_before)

    def test_reweighting_builds_a_new_decision(self, supplier):
        reweighted = _with_weights(supplier, [1.0, 1.0, 1.0, 1.0])
        assert np.allclose(reweighted.weights, 1.0)
        assert np.allclose(supplier.weights, [0.4, 0.3, 0.2, 0.1])


class TestRefusals:
    def test_one_option_has_no_stability_to_report(self):
        result = rank([[5.0]], [Criterion("A", 1.0)], labels=["Only"])
        with pytest.raises(McdaError, match="more than one option"):
            sensitivity(result)

    def test_it_takes_a_result_not_a_decision(self, supplier):
        with pytest.raises(McdaError, match="takes a Result"):
            sensitivity(supplier)
