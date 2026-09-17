"""Explaining a ranking, and the honesty of the explanation.

Two claims are under test. Where a method is additive, the reported
contributions must sum exactly to the score — an explanation whose parts do
not add up is not an explanation. Where it is not additive, the module must
say so rather than present an approximation as a decomposition.
"""

import numpy as np
import pytest

from mcdakit import METHODS, Criterion, Decision, McdaError, rank
from mcdakit.explain import ADDITIVE, compare, explain

CRITERIA = [
    Criterion("Price", 0.40, "cost", bounds=(2, 4)),
    Criterion("Quality", 0.25, "benefit", bounds=(0, 10)),
    Criterion("Lead time", 0.20, "cost", bounds=(5, 35)),
    Criterion("Support", 0.15, "benefit", bounds=(0, 10)),
]
MATRIX = [
    [2.75, 7.0, 14, 8.0],
    [2.90, 8.5, 16, 8.0],
    [3.40, 9.0, 11, 7.0],
    [2.20, 3.0, 32, 2.0],
]
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]


@pytest.fixture
def decision():
    return Decision(MATRIX, CRITERIA, LABELS)


class TestExactDecomposition:
    """For an additive method the parts must add up to the whole."""

    @pytest.mark.parametrize(
        "method", ["weighted_scoring", "saw", "simple_scoring", "spotis"]
    )
    def test_contributions_sum_to_the_score(self, decision, method):
        report = explain(rank(decision, method=method))
        total = sum(c.contribution for c in report.contributions)
        assert total == pytest.approx(report.score, abs=1e-9), (
            "an explanation whose parts do not sum to the score is not an explanation"
        )
        assert report.basis == "exact"

    def test_every_criterion_is_accounted_for(self, decision):
        report = explain(rank(decision, method="weighted_scoring"))
        assert {c.criterion for c in report.contributions} == {c.name for c in CRITERIA}

    def test_contributions_are_ordered_by_magnitude(self, decision):
        report = explain(rank(decision, method="weighted_scoring"))
        magnitudes = [abs(c.contribution) for c in report.contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert report.dominant == report.contributions[0].criterion

    def test_shares_are_proportions(self, decision):
        report = explain(rank(decision, method="weighted_scoring"))
        assert all(0.0 <= c.share <= 1.0 for c in report.contributions)
        assert sum(c.share for c in report.contributions) == pytest.approx(1.0)

    def test_a_muted_criterion_contributes_nothing(self):
        """A criterion weighted zero cannot have carried the result."""
        criteria = [
            Criterion("Real", 1.0, "benefit"),
            Criterion("Muted", 0.0, "benefit"),
        ]
        report = explain(rank([[1.0, 9.0], [9.0, 1.0]], criteria, labels=["A", "B"]))
        muted = next(c for c in report.contributions if c.criterion == "Muted")
        assert muted.contribution == pytest.approx(0.0)


class TestLeaveOneOutAttribution:
    """For a non-additive method, no exact decomposition exists."""

    @pytest.mark.parametrize("method", ["topsis", "vikor", "electre", "promethee"])
    def test_it_says_the_attribution_is_not_exact(self, decision, method):
        """Presenting an approximation as a decomposition would be the
        overstatement this package exists to avoid."""
        report = explain(rank(decision, method=method))
        assert report.basis == "leave_one_out"

    def test_the_printed_output_carries_the_caveat(self, decision):
        text = str(explain(rank(decision, method="topsis")))
        assert "do not sum to the score" in text

    def test_attribution_measures_the_effect_of_removal(self, decision):
        """The reported value must be the change caused by muting that
        criterion, not something merely correlated with it."""
        import dataclasses

        from mcdakit.ranking import score

        result = rank(decision, method="topsis")
        report = explain(result, "Option 1")
        index = LABELS.index("Option 1")
        baseline = float(result.scores[index])

        for contribution in report.contributions:
            position = [c.name for c in CRITERIA].index(contribution.criterion)
            weights = decision.weights.astype(float).copy()
            weights[position] = 0.0
            criteria = [
                dataclasses.replace(c, weight=float(w))
                for c, w in zip(decision.criteria, weights)
            ]
            without = Decision(decision.matrix, criteria, LABELS)
            scores, _f, _m = score(without, "topsis")
            assert contribution.contribution == pytest.approx(
                baseline - float(scores[index]), abs=1e-9
            )


class TestEveryMethod:
    @pytest.mark.parametrize("method", METHODS)
    def test_it_explains_every_shipped_method(self, decision, method):
        report = explain(rank(decision, method=method))
        assert len(report.contributions) == len(CRITERIA)
        assert report.basis in {"exact", "leave_one_out"}
        assert np.all(np.isfinite([c.contribution for c in report.contributions]))

    @pytest.mark.parametrize("method", METHODS)
    def test_the_basis_matches_the_methods_form(self, decision, method):
        report = explain(rank(decision, method=method))
        expected = "exact" if method in ADDITIVE else "leave_one_out"
        assert report.basis == expected


class TestCompare:
    def test_the_margin_matches_the_score_difference(self, decision):
        result = rank(decision, method="weighted_scoring")
        margin = compare(result, "Option 1", "Option 2")
        assert margin.margin == pytest.approx(
            result.score_of("Option 1") - result.score_of("Option 2")
        )

    def test_the_per_criterion_deltas_sum_to_the_margin(self, decision):
        """Only for an additive method, where the decomposition is exact."""
        margin = compare(
            rank(decision, method="weighted_scoring"), "Option 1", "Option 2"
        )
        assert sum(d for _c, d in margin.rows) == pytest.approx(margin.margin, abs=1e-9)

    def test_a_decisive_criterion_really_is_decisive(self, decision):
        """The claim is that removing this criterion's advantage reverses the
        outcome. Asserting the label without checking it would be worse than
        not reporting it."""
        margin = compare(
            rank(decision, method="weighted_scoring"), "Option 1", "Option 2"
        )
        if margin.decisive is None:
            pytest.skip("no single criterion is decisive on this problem")
        delta = dict(margin.rows)[margin.decisive]
        assert delta > margin.margin, (
            "a decisive criterion must contribute more than the whole margin"
        )

    def test_criteria_that_do_not_separate_are_labelled_as_such(self, decision):
        """A criterion the two options score alike favours neither."""
        text = str(
            compare(rank(decision, method="weighted_scoring"), "Option 1", "Option 2")
        )
        assert "no difference" in text

    def test_reversing_the_arguments_is_refused(self, decision):
        """Calling the loser the winner would silently invert every sign."""
        with pytest.raises(McdaError, match="did not outrank"):
            compare(rank(decision, method="weighted_scoring"), "Option 4", "Option 1")

    def test_it_works_for_non_additive_methods_too(self, decision):
        margin = compare(rank(decision, method="topsis"), "Option 2", "Option 1")
        assert margin.basis == "leave_one_out"
        assert margin.margin > 0


class TestRefusals:
    def test_an_unknown_option_is_refused(self, decision):
        with pytest.raises(McdaError, match="No option named"):
            explain(rank(decision), "Option 99")

    def test_it_takes_a_result_not_a_decision(self, decision):
        with pytest.raises(McdaError, match="takes a Result"):
            explain(decision)

    def test_it_defaults_to_explaining_the_winner(self, decision):
        result = rank(decision, method="weighted_scoring")
        assert explain(result).option == result.winner

    def test_a_single_criterion_problem_is_handled(self):
        """Muting the only criterion leaves nothing; the whole score rests
        on it."""
        criteria = [Criterion("Only", 1.0, "benefit")]
        report = explain(rank([[1.0], [9.0]], criteria, labels=["A", "B"]))
        assert len(report.contributions) == 1

    def test_a_single_criterion_problem_under_a_non_additive_method(self):
        criteria = [Criterion("Only", 1.0, "benefit")]
        report = explain(
            rank([[1.0], [9.0]], criteria, method="topsis", labels=["A", "B"])
        )
        assert report.contributions[0].contribution == pytest.approx(report.score)


class TestReadableOutput:
    def test_the_explanation_names_the_option_and_method(self, decision):
        text = str(explain(rank(decision, method="weighted_scoring")))
        assert "Option 1" in text and "weighted_scoring" in text
        assert "exact" in text

    def test_the_margin_names_both_options(self, decision):
        text = str(
            compare(rank(decision, method="weighted_scoring"), "Option 1", "Option 2")
        )
        assert "Option 1 beats Option 2" in text

    def test_a_contribution_prints_its_share(self, decision):
        report = explain(rank(decision, method="weighted_scoring"))
        assert "%" in str(report.contributions[0])


class TestNoSingleDecisiveCriterion:
    """A margin can rest on several criteria jointly rather than on one.

    The complement of the decisive case, and the more common one: reporting a
    criterion as decisive when none is would overstate how fragile the result
    is to a single disputed judgement.
    """

    def _wide_margin(self):
        """One option better on every criterion, so no single one carries it."""
        criteria = [
            Criterion("A", 0.5, "benefit"),
            Criterion("B", 0.5, "benefit"),
        ]
        return rank([[9.0, 9.0], [1.0, 1.0]], criteria, labels=["Strong", "Weak"])

    def test_no_criterion_is_reported_as_decisive(self):
        margin = compare(self._wide_margin(), "Strong", "Weak")
        assert margin.decisive is None, (
            "no single criterion exceeds a margin both of them build"
        )

    def test_the_printed_output_omits_the_decisive_line(self):
        text = str(compare(self._wide_margin(), "Strong", "Weak"))
        assert "decisive" not in text
        assert "Strong beats Weak" in text


class TestLongCriterionNamesStayReadable:
    """Real criterion names are often long ("Security & compliance
    certifications"). A fixed column width silently glues the name to the
    number, and explain() is the feature whose whole point is legibility.
    """

    def problem(self):
        criteria = [
            Criterion("Security & compliance certifications", 0.40, "benefit"),
            Criterion("Monthly infrastructure cost", 0.35, "cost"),
            Criterion("Support", 0.25, "benefit"),
        ]
        matrix = np.array([[8.0, 4200.0, 7.0], [6.0, 3800.0, 9.0], [9.0, 5000.0, 5.0]])
        return rank(
            matrix, criteria, method="weighted_scoring", labels=["AWS", "Azure", "GCP"]
        )

    def test_explain_separates_every_name_from_its_number(self):
        for line in str(explain(self.problem(), "AWS")).splitlines():
            if "+" in line or "-" in line:
                assert "  +" in line or "  -" in line, (
                    f"name and number are not separated: {line!r}"
                )

    def test_compare_separates_every_name_from_its_number(self):
        for line in str(compare(self.problem(), "AWS", "Azure")).splitlines():
            if "favours" in line:
                assert "  +" in line or "  -" in line, (
                    f"name and number are not separated: {line!r}"
                )
