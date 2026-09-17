"""Full AHP: pairwise comparison throughout the hierarchy.

Unlike every other method here, AHP takes no decision matrix. Alternatives are
compared against each other under each criterion, which is what makes it
usable on criteria nobody can measure — and what makes the judgements
checkable, since a person can contradict themselves in a way a column of
numbers cannot.
"""

import numpy as np
import pytest

from mcdakit import CONSISTENCY_LIMIT, McdaError, ahp_rank

#: Saaty's car-selection hierarchy, as reproduced in the AHP literature.
CRITERIA = np.array([[1.0, 1 / 2, 3.0], [2.0, 1.0, 4.0], [1 / 3, 1 / 4, 1.0]])
STYLE = np.array([[1.0, 1 / 4, 4.0], [4.0, 1.0, 4.0], [1 / 4, 1 / 4, 1.0]])
RELIABILITY = np.array([[1.0, 2.0, 5.0], [1 / 2, 1.0, 3.0], [1 / 5, 1 / 3, 1.0]])
FUEL = np.array([[1.0, 1 / 3, 1 / 2], [3.0, 1.0, 2.0], [2.0, 1 / 2, 1.0]])
NAMES = ["Style", "Reliability", "Fuel economy"]
LABELS = ["Civic", "Saturn", "Escort"]


def car_problem():
    return ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL], names=NAMES, labels=LABELS)


class TestTheHierarchy:
    def test_it_reproduces_saatys_car_example(self):
        """Values independently corroborated by `pyrepo-mcda`'s private
        `AHP._classic_ahp`, which agrees to fourteen significant figures."""
        result = car_problem()
        assert result.weights == pytest.approx([0.319618, 0.558425, 0.121957], abs=1e-6)
        assert result.scores == pytest.approx([0.426288, 0.443990, 0.129722], abs=1e-6)
        assert result.consistency["Style"] == pytest.approx(0.1873806698, abs=1e-9)
        assert [label for label, _ in result.ranking] == [
            "Saturn",
            "Civic",
            "Escort",
        ]

    def test_scores_are_priorities_summing_to_one(self):
        result = car_problem()
        assert result.scores.sum() == pytest.approx(1.0)
        assert np.all(result.scores > 0)

    def test_it_composes_local_priorities_with_criterion_weights(self):
        """The whole method in one line: alternatives-by-criteria multiplied
        by the criterion weights. Computed here independently."""
        from mcdakit.ahp import priorities

        weights = priorities(CRITERIA)
        local = np.column_stack([priorities(m) for m in (STYLE, RELIABILITY, FUEL)])
        assert car_problem().scores == pytest.approx(local @ weights, abs=1e-12)

    def test_the_criterion_weights_come_from_the_criteria_matrix(self):
        from mcdakit.ahp import priorities

        assert car_problem().weights == pytest.approx(priorities(CRITERIA), abs=1e-12)

    def test_local_priorities_are_alternatives_by_criteria(self):
        result = car_problem()
        assert result.local_priorities.shape == (3, 3)
        for column in range(3):
            assert result.local_priorities[:, column].sum() == pytest.approx(1.0)

    def test_a_perfectly_consistent_hierarchy_reports_zero(self):
        """3:1 and 9:1 imply 3:1 between the second and third."""
        consistent = np.array([[1.0, 3.0, 9.0], [1 / 3, 1.0, 3.0], [1 / 9, 1 / 3, 1.0]])
        result = ahp_rank(consistent, [consistent, consistent, consistent])
        assert all(
            r == pytest.approx(0.0, abs=1e-9) for r in result.consistency.values()
        )
        assert result.inconsistent == ()


class TestConsistencyIsReportedPerMatrix:
    """A hierarchy is only as sound as its least coherent judgement set, so an
    average would hide exactly what matters."""

    def test_every_matrix_gets_its_own_ratio(self):
        result = car_problem()
        assert set(result.consistency) == {"criteria", *NAMES}

    def test_an_inconsistent_matrix_is_named(self):
        result = car_problem()
        assert "Style" in result.inconsistent, (
            "the style judgements exceed Saaty's 0.10 limit and must be named"
        )
        assert result.consistency["Style"] > CONSISTENCY_LIMIT

    def test_the_coherent_matrices_are_not_named(self):
        result = car_problem()
        assert "Reliability" not in result.inconsistent
        assert "criteria" not in result.inconsistent

    def test_inconsistency_is_reported_never_enforced(self):
        """Whether a contradiction is fatal is the analyst's judgement. The
        ranking is still produced, and still usable."""
        result = car_problem()
        assert result.inconsistent
        assert result.scores.sum() == pytest.approx(1.0)
        assert result.winner in LABELS

    def test_the_printed_output_flags_the_offender(self):
        text = str(car_problem())
        assert "EXCEEDS 0.10" in text
        assert "Style contradicts itself" in text

    def test_a_coherent_hierarchy_prints_no_warning(self):
        consistent = np.array([[1.0, 3.0, 9.0], [1 / 3, 1.0, 3.0], [1 / 9, 1 / 3, 1.0]])
        text = str(ahp_rank(consistent, [consistent, consistent, consistent]))
        assert "EXCEEDS" not in text
        assert "contradict" not in text


class TestOrdering:
    def test_a_dominant_alternative_wins(self):
        """Preferred on every criterion, so no weighting can unseat it."""
        criteria = np.array([[1.0, 2.0], [1 / 2, 1.0]])
        dominant = np.array([[1.0, 7.0], [1 / 7, 1.0]])
        result = ahp_rank(criteria, [dominant, dominant], labels=["Strong", "Weak"])
        assert result.winner == "Strong"

    def test_criterion_weight_decides_between_opposed_alternatives(self):
        """Each alternative leads on one criterion, so the criterion weights
        settle it — which is the point of eliciting them separately."""
        favours_first = np.array([[1.0, 5.0], [1 / 5, 1.0]])
        favours_second = np.array([[1.0, 1 / 5], [5.0, 1.0]])

        first_matters = np.array([[1.0, 9.0], [1 / 9, 1.0]])
        second_matters = np.array([[1.0, 1 / 9], [9.0, 1.0]])

        a = ahp_rank(first_matters, [favours_first, favours_second], labels=["A", "B"])
        b = ahp_rank(second_matters, [favours_first, favours_second], labels=["A", "B"])
        assert a.winner == "A"
        assert b.winner == "B"

    def test_ranking_is_best_first(self):
        result = car_problem()
        scores = [s for _label, s in result.ranking]
        assert scores == sorted(scores, reverse=True)


class TestInput:
    def test_criteria_may_be_given_as_pairs(self):
        """The same shorthand ahp_weights accepts."""
        from_matrix = ahp_rank(
            np.array([[1.0, 3.0], [1 / 3, 1.0]]),
            [STYLE[:2, :2], RELIABILITY[:2, :2]],
        )
        from_pairs = ahp_rank({(0, 1): 3.0}, [STYLE[:2, :2], RELIABILITY[:2, :2]])
        assert from_pairs.scores == pytest.approx(from_matrix.scores)

    def test_names_and_labels_default_sensibly(self):
        result = ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL])
        assert result.names == ("Criterion 1", "Criterion 2", "Criterion 3")
        assert result.labels == ("Option 1", "Option 2", "Option 3")

    def test_no_comparison_matrices_is_refused(self):
        with pytest.raises(McdaError, match="one comparison matrix per"):
            ahp_rank(CRITERIA, [])

    def test_a_mismatched_criteria_matrix_is_refused(self):
        with pytest.raises(McdaError, match="one per criterion"):
            ahp_rank(CRITERIA, [STYLE, RELIABILITY])

    def test_matrices_comparing_different_alternatives_are_refused(self):
        """Every criterion must rank the same alternatives, or the columns
        cannot be combined."""
        with pytest.raises(McdaError, match="Every criterion must rank"):
            ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL[:2, :2]])

    def test_a_non_square_matrix_is_refused(self):
        with pytest.raises(McdaError, match="must be square"):
            ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL[:2, :]])

    def test_a_name_count_mismatch_is_refused(self):
        with pytest.raises(McdaError, match="criterion names"):
            ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL], names=["only one"])

    def test_a_label_count_mismatch_is_refused(self):
        with pytest.raises(McdaError, match="alternative labels"):
            ahp_rank(CRITERIA, [STYLE, RELIABILITY, FUEL], labels=["one", "two"])


class TestDistinctFromWhatOthersExposeAsAhp:
    """The *public* AHP of `pyrepo-mcda` takes a numeric decision matrix and
    returns a min-max weighted sum; `pyDecision`'s returns criterion weights
    from one comparison matrix, which is `ahp_weights` here. Neither compares
    alternatives pairwise. (`pyrepo-mcda` does implement full AHP privately,
    as `AHP._classic_ahp`; the values below were cross-checked against it and
    agree to fourteen significant figures.)"""

    def test_it_needs_no_decision_matrix(self):
        import inspect

        parameters = inspect.signature(ahp_rank).parameters
        assert "matrix" not in parameters
        assert "criteria_comparisons" in parameters
        assert "alternative_comparisons" in parameters

    def test_criterion_weights_agree_with_ahp_weights(self):
        """The two share a derivation, and should not drift apart."""
        from mcdakit import ahp_weights

        elicited = ahp_weights(CRITERIA, names=NAMES)
        assert car_problem().weights == pytest.approx(elicited["weights"], abs=1e-12)
