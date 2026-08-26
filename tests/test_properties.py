"""Property-based tests: invariants that must hold on any input.

The worked examples check that specific answers are right. These check that
whole classes of transformation leave the answer alone — the kind of bug a
fixture cannot catch because it only ever sees one arrangement of the data.
"""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from mcdakit import METHODS, Criterion, Decision, rank

# Ranking is not cheap, and PROMETHEE and ELECTRE are O(n^2) in Python loops.
# Keep the generated problems small and the example count modest.
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

values = st.floats(min_value=0.1, max_value=100.0, allow_nan=False,
                   allow_infinity=False)
weights = st.floats(min_value=0.01, max_value=10.0, allow_nan=False,
                    allow_infinity=False)


@st.composite
def problems(draw, min_options=2, max_options=5, max_criteria=4):
    n_options = draw(st.integers(min_options, max_options))
    n_criteria = draw(st.integers(1, max_criteria))
    matrix = draw(
        st.lists(
            st.lists(values, min_size=n_criteria, max_size=n_criteria),
            min_size=n_options, max_size=n_options,
        )
    )
    criteria = [
        Criterion(f"C{j + 1}", draw(weights), "benefit", bounds=(0.0, 200.0))
        for j in range(n_criteria)
    ]
    return Decision(matrix, criteria)


class TestScaleInvariance:
    @given(problem=problems())
    @SETTINGS
    @pytest.mark.parametrize("method", METHODS)
    def test_scaling_all_weights_leaves_the_order_alone(self, problem, method):
        """Weights are relative. Doubling all of them is the same input."""
        doubled = [
            Criterion(c.name, c.weight * 2, c.direction, c.bounds)
            for c in problem.criteria
        ]
        scaled = Decision(problem.matrix, doubled, problem.labels)
        assert rank(scaled, method=method).order == rank(
            problem, method=method).order


class TestPermutationInvariance:
    @given(problem=problems())
    @SETTINGS
    @pytest.mark.parametrize("method", METHODS)
    def test_reordering_the_options_does_not_change_who_wins(
        self, problem, method
    ):
        """The row order of the matrix is presentation, not information.

        Ties are exempt: which of two equal options is listed first is
        genuinely arbitrary, so only a strict winner is asserted on.
        """
        original = rank(problem, method=method)
        scores = np.sort(original.scores)
        assume(len(scores) < 2 or scores[-1] - scores[-2] > 1e-9)

        permutation = list(reversed(range(problem.n_options)))
        shuffled = Decision(
            problem.matrix[permutation, :],
            problem.criteria,
            [problem.labels[i] for i in permutation],
        )
        assert rank(shuffled, method=method).winner == original.winner

    @given(problem=problems())
    @SETTINGS
    @pytest.mark.parametrize("method", METHODS)
    def test_reordering_the_criteria_does_not_change_who_wins(
        self, problem, method
    ):
        """Columns carry their own weight and direction, so their order is
        presentation too."""
        original = rank(problem, method=method)
        scores = np.sort(original.scores)
        assume(len(scores) < 2 or scores[-1] - scores[-2] > 1e-9)

        permutation = list(reversed(range(problem.n_criteria)))
        shuffled = Decision(
            problem.matrix[:, permutation],
            [problem.criteria[j] for j in permutation],
            problem.labels,
        )
        assert rank(shuffled, method=method).winner == original.winner


class TestDominance:
    @given(problem=problems())
    @SETTINGS
    @pytest.mark.parametrize("method", METHODS)
    def test_an_option_better_on_everything_wins(self, problem, method):
        """The one result no method may get wrong. Weighted `simple_scoring`
        included: it ignores weights but still sums the raw values."""
        dominant = problem.matrix.max(axis=0) + 1.0
        extended = Decision(
            np.vstack([problem.matrix, dominant]),
            problem.criteria,
            [*problem.labels, "Dominant"],
        )
        assert rank(extended, method=method).winner == "Dominant"


class TestOutputShape:
    @given(problem=problems())
    @SETTINGS
    @pytest.mark.parametrize("method", METHODS)
    def test_every_option_is_scored_finitely_and_ranked_once(
        self, problem, method
    ):
        result = rank(problem, method=method)
        assert result.scores.shape == (problem.n_options,)
        assert np.all(np.isfinite(result.scores))
        assert len(result.ranking) == problem.n_options
        assert sorted(label for label, _ in result.ranking) == sorted(
            problem.labels)
        assert min(result.ranks) == 1


class TestSpotisProperty:
    @given(problem=problems(min_options=3))
    @SETTINGS
    def test_removing_any_loser_never_reorders_the_rest(self, problem):
        """The reversal-freedom guarantee, over arbitrary generated problems
        rather than the handful in the benchmark."""
        full = rank(problem, method="spotis")
        for index, _label in enumerate(problem.labels):
            if index == full.winner_index:
                continue
            reduced = rank(problem.without(index), method="spotis")
            for kept in reduced.decision.labels:
                assert reduced.score_of(kept) == pytest.approx(
                    full.score_of(kept), abs=1e-9)
