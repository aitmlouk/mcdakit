"""Decisions made by several people.

The claim these tests hold to account: a consensus ranking that hides
disagreement is the same failure as a point estimate that hides its error
bars. Every test here is either about combining views correctly, or about
refusing to combine views that cannot honestly be combined.
"""

import numpy as np
import pytest

from mcdakit import Criterion, McdaError
from mcdakit.group import (
    Participant,
    aggregate_rankings,
    aggregate_scores,
    disagreement,
    group_rank,
)


@pytest.fixture
def criteria():
    return [Criterion("Price", 0.5, "cost"), Criterion("Quality", 0.5)]


@pytest.fixture
def agreeing():
    """Two people who broadly agree: A is cheaper, B is better."""
    return [
        Participant("Alice", [[2.0, 8.0], [3.0, 6.0]]),
        Participant("Bob", [[2.0, 7.0], [3.0, 6.5]]),
    ]


@pytest.fixture
def split():
    """Three people where one holds an extreme view on a single option."""
    return [
        Participant("Zealot", [[100.0], [5.0], [6.0]]),
        Participant("Ann", [[1.0], [7.0], [8.0]]),
        Participant("Bo", [[1.0], [7.0], [9.0]]),
    ]


class TestAggregateScores:
    def test_it_averages_cell_by_cell(self, criteria):
        people = [
            Participant("A", [[2.0, 8.0]]),
            Participant("B", [[4.0, 6.0]]),
        ]
        decision = aggregate_scores(people, criteria)
        assert decision.matrix.ravel() == pytest.approx([3.0, 7.0])

    def test_participant_weight_shifts_the_average(self, criteria):
        """A chair with a casting vote is expressed by weight, not by being
        entered twice."""
        people = [
            Participant("Chair", [[2.0, 8.0]], weight=3.0),
            Participant("Member", [[6.0, 4.0]], weight=1.0),
        ]
        decision = aggregate_scores(people, criteria)
        assert decision.matrix.ravel() == pytest.approx([3.0, 7.0])

    def test_equal_weights_are_the_default(self, criteria, agreeing):
        weighted = [Participant(p.name, p.matrix, weight=5.0) for p in agreeing]
        assert aggregate_scores(agreeing, criteria).matrix.ravel() == (
            pytest.approx(aggregate_scores(weighted, criteria).matrix.ravel())
        )


class TestAggregateRankings:
    def test_unanimous_participants_give_that_order(self, criteria, agreeing):
        out = aggregate_rankings(agreeing, criteria, labels=["A", "B"])
        assert out["order"] == ["A", "B"]

    def test_each_participant_is_ranked_separately(self, criteria, agreeing):
        out = aggregate_rankings(agreeing, criteria, labels=["A", "B"])
        assert set(out["per_participant"]) == {"Alice", "Bob"}
        assert all(r.winner == "A" for r in out["per_participant"].values())

    def test_it_ignores_magnitude_where_averaging_does_not(self, split):
        """The point of having both: one wild score decides the mean but
        carries no more weight than any other vote in the count."""
        criteria = [Criterion("Q", 1.0)]
        borda = aggregate_rankings(split, criteria, labels=["A", "B", "C"])
        mean = aggregate_scores(split, criteria, labels=["A", "B", "C"])
        assert borda["order"][0] == "C"
        assert int(np.argmax(mean.matrix)) == 0, "the outlier should dominate the mean"

    def test_ties_do_not_invent_a_preference(self):
        """Two options nobody separated must score equally, rather than being
        ordered by however they were listed."""
        criteria = [Criterion("Q", 1.0)]
        people = [
            Participant("A", [[5.0], [5.0], [1.0]]),
            Participant("B", [[5.0], [5.0], [1.0]]),
        ]
        out = aggregate_rankings(people, criteria, labels=["X", "Y", "Z"])
        assert out["scores"][0] == pytest.approx(out["scores"][1])


class TestDisagreement:
    def test_full_agreement_has_zero_spread(self, criteria):
        matrix = [[2.0, 8.0], [3.0, 6.0]]
        people = [Participant("A", matrix), Participant("B", matrix)]
        assert disagreement(people, criteria)["overall"] == pytest.approx(0.0)

    def test_it_distinguishes_a_shared_mean_from_a_contested_one(self):
        """5.5 that everyone agreed on and 5.5 that averages 9 and 2 are the
        same number and completely different findings."""
        criteria = [Criterion("Q", 1.0)]
        calm = [Participant("A", [[5.5]]), Participant("B", [[5.5]])]
        fought = [Participant("A", [[9.0]]), Participant("B", [[2.0]])]

        assert aggregate_scores(calm, criteria).matrix.ravel() == (
            pytest.approx(aggregate_scores(fought, criteria).matrix.ravel())
        )
        assert disagreement(calm, criteria)["overall"] == pytest.approx(0.0)
        assert disagreement(fought, criteria)["overall"] > 3.0

    def test_it_names_the_most_contested_option_and_criterion(self):
        criteria = [Criterion("Calm", 1.0), Criterion("Contested", 1.0)]
        people = [
            Participant("A", [[5.0, 1.0], [5.0, 5.0]]),
            Participant("B", [[5.0, 9.0], [5.0, 5.0]]),
        ]
        report = disagreement(people, criteria, labels=["Fought", "Settled"])
        assert report["most_contested_option"] == "Fought"
        assert report["most_contested_criterion"] == "Contested"

    def test_spread_is_unweighted(self):
        """How far the people differ, not how much the chair outweighs them —
        weighting the spread would hide a minority objection."""
        criteria = [Criterion("Q", 1.0)]
        plain = [Participant("A", [[9.0]]), Participant("B", [[1.0]])]
        skewed = [
            Participant("A", [[9.0]], weight=99.0),
            Participant("B", [[1.0]], weight=1.0),
        ]
        assert disagreement(plain, criteria)["overall"] == pytest.approx(
            disagreement(skewed, criteria)["overall"]
        )


class TestGroupRank:
    def test_it_reports_both_aggregations(self, criteria, agreeing):
        out = group_rank(agreeing, criteria, labels=["A", "B"])
        assert out["result"].winner == "A"
        assert out["borda"]["order"][0] == "A"
        assert out["agree"] is True
        assert out["unanimous"] is True

    def test_when_the_two_aggregations_disagree_it_says_so(self, split):
        """The finding the module exists for: the answer depends on whether
        the participants are co-estimators or voters."""
        criteria = [Criterion("Q", 1.0)]
        out = group_rank(split, criteria, labels=["A", "B", "C"])
        assert out["agree"] is False
        assert out["result"].winner != out["borda"]["order"][0]

    def test_it_reports_when_participants_individually_disagree(self):
        criteria = [Criterion("Q", 1.0)]
        people = [
            Participant("A", [[9.0], [1.0]]),
            Participant("B", [[1.0], [9.0]]),
        ]
        assert group_rank(people, criteria, labels=["X", "Y"])["unanimous"] is False

    def test_every_participant_gets_their_own_result(self, criteria, agreeing):
        out = group_rank(agreeing, criteria, labels=["A", "B"])
        assert set(out["per_participant"]) == {"Alice", "Bob"}

    def test_it_works_with_any_method(self, criteria, agreeing):
        for method in ("topsis", "vikor", "electre", "promethee"):
            out = group_rank(agreeing, criteria, method=method, labels=["A", "B"])
            assert out["result"].method == method

    def test_spotis_keeps_its_guarantee_on_the_averaged_matrix(self):
        """Averaging scores does not touch the bounds, so reversal-freedom
        survives aggregation."""
        criteria = [
            Criterion("Price", 0.5, "cost", bounds=(0, 10)),
            Criterion("Quality", 0.5, "benefit", bounds=(0, 10)),
        ]
        people = [
            Participant("A", [[2.0, 8.0], [3.0, 6.0], [9.0, 1.0]]),
            Participant("B", [[2.5, 7.0], [3.0, 6.5], [8.0, 2.0]]),
        ]
        out = group_rank(people, criteria, method="spotis", labels=["X", "Y", "Z"])
        assert out["result"].reversal_free is True


class TestRefusals:
    def test_different_option_sets_are_refused(self, criteria):
        """The mistake this prevents: an option only one person scored gets
        averaged over one opinion, then ranked against options averaged over
        everybody — authoritative-looking, computed from uneven evidence."""
        people = [
            Participant("A", [[2.0, 8.0], [3.0, 6.0]]),
            Participant("B", [[2.0, 8.0]]),
        ]
        with pytest.raises(McdaError, match="not scoring the same problem"):
            aggregate_scores(people, criteria)

    def test_different_criteria_counts_are_refused(self, criteria):
        people = [
            Participant("A", [[2.0, 8.0]]),
            Participant("B", [[2.0, 8.0, 1.0]]),
        ]
        with pytest.raises(McdaError, match="not scoring the same problem"):
            aggregate_scores(people, criteria)

    def test_a_criterion_count_mismatch_is_refused(self):
        people = [Participant("A", [[1.0, 2.0]]), Participant("B", [[3.0, 4.0]])]
        with pytest.raises(McdaError, match="but 1 were given"):
            aggregate_scores(people, [Criterion("Only", 1.0)])

    def test_a_single_participant_is_refused(self, criteria):
        with pytest.raises(McdaError, match="at least two participants"):
            aggregate_scores([Participant("Alone", [[1.0, 2.0]])], criteria)

    def test_no_participants_is_refused(self, criteria):
        with pytest.raises(McdaError, match="at least one participant"):
            aggregate_scores([], criteria)

    def test_all_zero_participant_weights_are_refused(self, criteria):
        people = [
            Participant("A", [[1.0, 2.0]], weight=0.0),
            Participant("B", [[3.0, 4.0]], weight=0.0),
        ]
        with pytest.raises(McdaError, match="all zero"):
            aggregate_scores(people, criteria)

    def test_a_nameless_participant_is_refused(self):
        with pytest.raises(McdaError, match="needs a name"):
            Participant("  ", [[1.0]])

    def test_a_negative_participant_weight_is_refused(self):
        with pytest.raises(McdaError, match="non-negative"):
            Participant("A", [[1.0]], weight=-1.0)
