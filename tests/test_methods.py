"""The eight methods, against values verified outside this codebase.

Expected numbers come from the Odoo module's test suite, where they were hand
computed, or from the textbook definition of the method.
"""

import numpy as np
import pytest

from mcdakit import METHODS, Criterion, Decision, McdaError, rank
from mcdakit.methods import preference_degree
from mcdakit.orientation import orient


class TestWeightedScoring:
    def test_matches_the_hand_computation(self, supplier):
        """The headline number must be arithmetically right.

        Min-max normalise each criterion across the options, then weight:
        B = 0.6000, A = 0.5250, C = 0.2833.
        """
        result = rank(supplier, method="weighted_scoring")
        assert result.order == ["Supplier B", "Supplier A", "Supplier C"]
        assert result.score_of("Supplier B") == pytest.approx(0.6000, abs=1e-4)
        assert result.score_of("Supplier A") == pytest.approx(0.5250, abs=1e-4)
        assert result.score_of("Supplier C") == pytest.approx(0.2833, abs=1e-4)
        assert list(result.ranks) == [2, 1, 3]

    def test_a_flat_criterion_does_not_divide_by_zero(self, supplier_criteria):
        """Every option scoring the same carries no information; it must
        contribute a constant, not a NaN."""
        matrix = [[5, 5, 7, 6], [5, 9, 8, 9], [5, 7, 6, 5]]
        scores = rank(matrix, supplier_criteria).scores
        assert np.all(np.isfinite(scores))


class TestTopsis:
    def test_scores_are_closeness_coefficients(self, supplier):
        result = rank(supplier, method="topsis")
        assert np.all((result.scores >= 0) & (result.scores <= 1))

    def test_a_dominant_option_scores_one(self, supplier_criteria):
        """An option best on every criterion sits on the ideal point."""
        matrix = [[9, 9, 9, 9], [1, 2, 3, 4], [5, 5, 5, 5]]
        result = rank(
            matrix,
            supplier_criteria,
            method="topsis",
            labels=["Best", "Worst", "Middle"],
        )
        assert result.winner == "Best"
        assert result.score_of("Best") == pytest.approx(1.0)
        assert result.score_of("Worst") == pytest.approx(0.0)


class TestVikor:
    def test_scores_are_negated_so_higher_is_better(self, supplier):
        """VIKOR's native Q is smallest-is-best; we return -Q, so scores lie
        in [-1, 0]. The maximum is 0 only when one option is best on both the
        group-utility and the individual-regret measure, which need not
        happen — here Supplier B wins on S while C ties it on R."""
        result = rank(supplier, method="vikor")
        assert np.all((result.scores >= -1 - 1e-12) & (result.scores <= 1e-12))

    def test_worst_case_regret_can_outvote_group_utility(self, supplier):
        """VIKOR picks Supplier A where every additive method picks B, and
        that is correct, not a porting bug.

        B has the best group utility (S = 0.400 against A's 0.475) but the
        worst single-criterion regret (R = 0.400: it is last on Price, which
        carries 40% of the weight). At the default v = 0.5 the regret penalty
        outweighs the utility edge. Leaning on group utility alone restores B.
        """
        assert rank(supplier, method="vikor").winner == "Supplier A"
        assert rank(supplier, method="weighted_scoring").winner == "Supplier B"
        assert rank(supplier, method="vikor", v=1.0).winner == "Supplier B"

    def test_v_trades_group_utility_against_worst_case(self, supplier):
        """v=1 uses group utility alone, v=0 the worst single criterion. They
        need not agree, and the parameter must actually reach the method."""
        utility = rank(supplier, method="vikor", v=1.0).scores
        regret = rank(supplier, method="vikor", v=0.0).scores
        assert not np.allclose(utility, regret)


class TestElectre:
    def test_scores_are_net_outranking_counts(self, supplier):
        result = rank(supplier, method="electre")
        assert np.allclose(result.scores, np.round(result.scores))
        assert result.scores.sum() == pytest.approx(0.0), (
            "every outranking is one option's gain and another's loss"
        )


class TestSaw:
    def test_normalises_by_the_column_maximum(self, supplier_criteria):
        """r_ij = x_ij / max_j, so the best option on a criterion scores its
        full weight there."""
        matrix = [[10, 10, 10, 10], [5, 5, 5, 5]]
        scores = rank(matrix, supplier_criteria, method="saw").scores
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.5)


class TestSimpleScoring:
    def test_ignores_weights_entirely(self, supplier):
        """Documented behaviour, and the reason it is a baseline only."""
        heavy = [
            Criterion(c.name, weight=99 if c.name == "Price" else 0.01)
            for c in supplier.criteria
        ]
        reweighted = Decision(supplier.matrix, heavy, supplier.labels)
        assert np.allclose(
            rank(supplier, method="simple_scoring").scores,
            rank(reweighted, method="simple_scoring").scores,
        )
        assert rank(supplier, method="simple_scoring").scores[0] == 27


class TestPreferenceDegree:
    """The six shapes against their textbook definitions."""

    def test_usual_treats_any_difference_as_total(self):
        assert preference_degree(0.0001, "usual") == 1.0
        assert preference_degree(0, "usual") == 0.0

    def test_ushape_ignores_differences_up_to_q(self):
        assert preference_degree(10, "ushape", q=10) == 0.0
        assert preference_degree(11, "ushape", q=10) == 1.0

    def test_vshape_grows_linearly_to_p(self):
        assert preference_degree(5, "vshape", p=20) == pytest.approx(0.25)
        assert preference_degree(10, "vshape", p=20) == pytest.approx(0.50)
        assert preference_degree(25, "vshape", p=20) == 1.0

    def test_level_steps_through_a_half(self):
        assert preference_degree(5, "level", q=10, p=20) == 0.0
        assert preference_degree(15, "level", q=10, p=20) == 0.5
        assert preference_degree(25, "level", q=10, p=20) == 1.0

    def test_linear_interpolates_between_q_and_p(self):
        assert preference_degree(10, "linear", q=10, p=20) == 0.0
        assert preference_degree(15, "linear", q=10, p=20) == pytest.approx(0.5)
        assert preference_degree(20, "linear", q=10, p=20) == 1.0

    def test_gaussian_reaches_its_known_value_at_s(self):
        """1 - exp(-1/2) is about 0.3935 at d = s."""
        assert preference_degree(20, "gaussian", s=20) == pytest.approx(
            0.3935, abs=1e-4
        )

    @pytest.mark.parametrize(
        "shape", ["usual", "ushape", "vshape", "level", "linear", "gaussian"]
    )
    def test_a_worse_option_is_never_preferred(self, shape):
        assert preference_degree(-5, shape, q=1, p=2, s=3) == 0.0


class TestPromethee:
    def test_net_flows_stay_within_minus_one_and_one(self, supplier_criteria):
        """Normalising by the weight total keeps flows on PROMETHEE's scale
        whatever units the weights were entered in."""
        big = [Criterion(c.name, c.weight * 100) for c in supplier_criteria]
        matrix = [[9, 5, 7, 6], [6, 9, 8, 9], [7, 7, 6, 5]]
        scores = rank(matrix, big, method="promethee").scores
        assert np.all(np.abs(scores) <= 1.0)

    def test_the_default_shape_gives_the_expected_order(self, supplier):
        assert rank(supplier, method="promethee").order == [
            "Supplier B",
            "Supplier A",
            "Supplier C",
        ]

    def test_a_negligible_difference_stops_deciding_the_outcome(self):
        """The point of preference functions: one euro should not outweigh
        three points of quality."""
        matrix = [[4200, 6], [4201, 9]]
        labels = ["Cheap", "Good"]

        usual = [Criterion("Price", 0.5, "cost"), Criterion("Quality", 0.5)]
        tied = rank(matrix, usual, method="promethee", labels=labels)
        assert tied.scores[0] == pytest.approx(tied.scores[1]), (
            "with the usual criterion the 1-unit price edge cancels the "
            "3-point quality gap"
        )

        shrug = [
            Criterion("Price", 0.5, "cost", preference_shape="linear", q=100, p=500),
            Criterion("Quality", 0.5),
        ]
        assert rank(matrix, shrug, method="promethee", labels=labels).winner == "Good"


class TestOrientation:
    def test_cost_columns_are_mirrored(self):
        """max + min - x mirrors the column without squashing it, so the gaps
        between options stay proportional."""
        data = np.array([[100.0], [400.0], [900.0]])
        flipped = orient(data, ["cost"])
        assert flipped[0, 0] == pytest.approx(900.0)
        assert flipped[2, 0] == pytest.approx(100.0)
        assert flipped.min() >= 0.0, "TOPSIS and SAW rely on non-negative values"

    def test_benefit_columns_are_untouched(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(orient(data, ["benefit", "benefit"]), data)

    @pytest.mark.parametrize("method", METHODS)
    def test_real_figures_rank_correctly_for_every_method(self, method):
        """Without direction handling, a buyer entering real figures into
        "Price (EUR)" and "Delivery (days)" gets the most expensive, slowest
        option ranked first: a confident, inverted answer."""
        criteria = [
            Criterion("Price (EUR)", 0.5, "cost", bounds=(0, 1000)),
            Criterion("Delivery (days)", 0.5, "cost", bounds=(0, 60)),
        ]
        matrix = [[100.0, 2.0], [900.0, 30.0]]
        labels = ["Cheap and fast", "Pricey and slow"]

        if method == "simple_scoring":
            pytest.skip("ignores weights and magnitudes by design")
        result = rank(matrix, criteria, method=method, labels=labels)
        assert result.winner == "Cheap and fast", (
            f"{method} ranked the expensive, slow supplier first"
        )

    def test_a_benefit_criterion_is_not_inverted(self):
        criteria = [Criterion("Capacity", 1.0, "benefit")]
        result = rank([[100.0], [900.0]], criteria, labels=["Small", "Large"])
        assert result.winner == "Large"


class TestEveryMethod:
    @pytest.mark.parametrize("method", METHODS)
    def test_produces_a_complete_ranking(self, procurement, method):
        result = rank(procurement, method=method)
        assert len(result.ranking) == procurement.n_options
        assert min(result.ranks) == 1
        assert np.all(np.isfinite(result.scores))

    @pytest.mark.parametrize("method", METHODS)
    def test_a_single_option_is_trivially_first(self, method):
        criteria = [Criterion("A", 1.0, bounds=(0, 10))]
        result = rank([[5.0]], criteria, method=method, labels=["Only"])
        assert result.winner == "Only"

    def test_an_unknown_method_lists_the_real_ones(self, supplier):
        with pytest.raises(McdaError, match="Unknown method"):
            rank(supplier, method="ahp")
