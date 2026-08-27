"""Language-model assistance, and the checks that make it usable.

The design claim under test: the model proposes, the package verifies. Every
test here is either about refusing a bad suggestion, or about telling the
caller what a suggestion is worth. No test contacts a model; the provider is
a plain callable, which is what makes the module testable at all.
"""

import pytest

from mcdakit import Criterion, McdaError
from mcdakit.ai import (
    AiError,
    critique_weights,
    propose_criteria,
    propose_weights,
)

MATRIX = [[2.75, 7.0, 14], [2.90, 8.5, 16], [3.40, 9.0, 11], [2.20, 3.0, 32]]


@pytest.fixture
def criteria():
    return [
        Criterion("Price", 1.0, "cost", bounds=(2, 4)),
        Criterion("Quality", 1.0, "benefit", bounds=(0, 10)),
        Criterion("Lead time", 1.0, "cost", bounds=(5, 35)),
    ]


def replying(*texts):
    """A provider that returns the given replies in turn, then repeats."""
    state = {"n": 0}

    def ask(_prompt):
        reply = texts[state["n"] % len(texts)]
        state["n"] += 1
        return reply

    return ask


GOOD_CRITERIA = """[
  {"name": "Price", "direction": "cost", "weight": 0.5, "bounds": [2, 4]},
  {"name": "Quality", "direction": "benefit", "weight": 0.5, "bounds": [0, 10]}
]"""


class TestProposeCriteria:
    def test_it_returns_validated_criteria(self):
        proposal = propose_criteria("a supplier choice", ask=replying(GOOD_CRITERIA))
        assert [c.name for c in proposal.criteria] == ["Price", "Quality"]
        assert proposal.criteria[0].direction == "cost"
        assert proposal.criteria[0].bounds == (2.0, 4.0)

    def test_nothing_is_applied_automatically(self):
        """A proposal is a suggestion. Applying it is the caller's decision,
        and the flag says so rather than leaving it implicit."""
        proposal = propose_criteria("x", ask=replying(GOOD_CRITERIA))
        assert proposal.accepted is False

    def test_json_wrapped_in_prose_and_fences_is_recovered(self):
        """Models add commentary however firmly they are told not to."""
        reply = (
            "Certainly! Here are the criteria:\n"
            f"```json\n{GOOD_CRITERIA}\n```\n"
            "Hope this helps."
        )
        proposal = propose_criteria("x", ask=replying(reply))
        assert len(proposal.criteria) == 2

    def test_an_invalid_suggestion_is_discarded_with_its_reason(self):
        """The count matters: four proposed of which two were malformed is a
        different signal from two proposed."""
        reply = """[
          {"name": "Price", "direction": "cost", "weight": 0.5},
          {"name": "Bad direction", "direction": "sideways", "weight": 0.5},
          {"name": "Negative", "direction": "cost", "weight": -1}
        ]"""
        proposal = propose_criteria("x", ask=replying(reply))
        assert [c.name for c in proposal.criteria] == ["Price"]
        assert len(proposal.rejected) == 2
        names = [n for n, _why in proposal.rejected]
        assert "Bad direction" in names and "Negative" in names
        assert all(why for _n, why in proposal.rejected), "each needs a reason"

    def test_a_reply_with_no_valid_criterion_is_refused(self):
        reply = '[{"name": "Bad", "direction": "sideways", "weight": 1}]'
        with pytest.raises(AiError, match="No criterion"):
            propose_criteria("x", ask=replying(reply))

    def test_a_reply_containing_no_json_is_refused(self):
        with pytest.raises(AiError, match="no JSON"):
            propose_criteria("x", ask=replying("I cannot help with that."))

    def test_malformed_json_is_refused(self):
        with pytest.raises(AiError, match="not valid JSON"):
            propose_criteria("x", ask=replying('[{"name": "Price",}]'))

    def test_a_json_object_where_a_list_was_required_is_refused(self):
        with pytest.raises(AiError, match="Expected a JSON list"):
            propose_criteria("x", ask=replying('{"name": "Price"}'))


class TestSelfConsistency:
    def test_a_consistent_model_scores_high_agreement(self):
        proposal = propose_criteria("x", ask=replying(GOOD_CRITERIA), samples=3)
        assert proposal.agreement == pytest.approx(1.0)

    def test_an_improvising_model_is_exposed(self):
        """The check that a single reply cannot provide: a model naming
        different criteria each time is guessing, and the caller should see
        that rather than the first answer alone."""
        other = """[
          {"name": "Colour", "direction": "benefit", "weight": 0.5},
          {"name": "Mood", "direction": "benefit", "weight": 0.5}
        ]"""
        proposal = propose_criteria("x", ask=replying(GOOD_CRITERIA, other), samples=2)
        assert proposal.agreement == pytest.approx(0.0)

    def test_a_single_sample_reports_no_agreement(self):
        """With one reply there is nothing to compare, and inventing a score
        would imply a check that did not happen."""
        assert propose_criteria("x", ask=replying(GOOD_CRITERIA)).agreement is None

    def test_zero_samples_is_refused(self):
        with pytest.raises(McdaError, match="at least 1"):
            propose_criteria("x", ask=replying(GOOD_CRITERIA), samples=0)


class TestProposeWeights:
    GOOD = '{"Price": 0.5, "Quality": 0.3, "Lead time": 0.2}'

    def test_weights_are_normalised(self, criteria):
        proposal = propose_weights(
            "x", criteria, ask=replying('{"Price": 5, "Quality": 3, "Lead time": 2}')
        )
        assert proposal.weights.sum() == pytest.approx(1.0)
        assert proposal.weights == pytest.approx([0.5, 0.3, 0.2])

    def test_a_missing_criterion_names_the_one_omitted(self, criteria):
        with pytest.raises(AiError, match="omitted a weight"):
            propose_weights(
                "x", criteria, ask=replying('{"Price": 0.5, "Quality": 0.5}')
            )

    def test_a_negative_weight_is_refused(self, criteria):
        """A criterion cannot count against itself, whatever proposed it."""
        with pytest.raises(AiError, match="negative or non-finite"):
            propose_weights(
                "x",
                criteria,
                ask=replying('{"Price": -1, "Quality": 1, "Lead time": 1}'),
            )

    def test_all_zero_weights_are_refused(self, criteria):
        with pytest.raises(AiError, match="summing to zero"):
            propose_weights(
                "x",
                criteria,
                ask=replying('{"Price": 0, "Quality": 0, "Lead time": 0}'),
            )

    def test_a_non_numeric_weight_is_refused(self, criteria):
        with pytest.raises(AiError, match="not a number"):
            propose_weights(
                "x",
                criteria,
                ask=replying('{"Price": "high", "Quality": 1, "Lead time": 1}'),
            )

    def test_a_list_where_an_object_was_required_is_refused(self, criteria):
        with pytest.raises(AiError, match="Expected a JSON object"):
            propose_weights("x", criteria, ask=replying("[0.5, 0.3, 0.2]"))


class TestVerificationAgainstData:
    """The reason this module exists rather than a bare LLM call."""

    GOOD = '{"Price": 0.5, "Quality": 0.3, "Lead time": 0.2}'

    def test_the_proposal_is_checked_for_stability(self, criteria):
        """A model's weights may be plausible and still yield a
        recommendation that a small correction overturns."""
        proposal = propose_weights(
            "x", criteria, ask=replying(self.GOOD), matrix=MATRIX
        )
        assert proposal.stability
        assert proposal.stability["level"] in {
            "fragile",
            "moderate",
            "robust",
            "immovable",
        }

    def test_divergence_from_objective_schemes_is_reported(self, criteria):
        """A model that orders importance unlike every data-driven scheme is
        not necessarily wrong, but the caller should know."""
        contrarian = '{"Price": 0.98, "Quality": 0.01, "Lead time": 0.01}'
        proposal = propose_weights(
            "x", criteria, ask=replying(contrarian), matrix=MATRIX
        )
        assert proposal.disagrees_with, "expected disagreement to be surfaced"

    def test_no_matrix_means_no_verification_claimed(self, criteria):
        """Without data there is nothing to check against, and an empty
        report is honest where a fabricated one would not be."""
        proposal = propose_weights("x", criteria, ask=replying(self.GOOD))
        assert proposal.stability == {}
        assert proposal.disagrees_with == ()

    def test_weight_self_consistency_is_measured(self, criteria):
        other = '{"Price": 0.1, "Quality": 0.8, "Lead time": 0.1}'
        proposal = propose_weights(
            "x", criteria, ask=replying(self.GOOD, other), samples=2
        )
        assert 0.0 <= proposal.agreement <= 1.0

    def test_zero_samples_is_refused(self, criteria):
        with pytest.raises(McdaError, match="at least 1"):
            propose_weights("x", criteria, ask=replying(self.GOOD), samples=0)


class TestCritiqueWeights:
    """The same scrutiny, applied to weights from any source."""

    def test_it_assesses_weights_the_package_never_proposed(self, criteria):
        out = critique_weights([0.5, 0.3, 0.2], criteria, MATRIX)
        assert out["weights"].sum() == pytest.approx(1.0)
        assert out["stability"]["level"] in {
            "fragile",
            "moderate",
            "robust",
            "immovable",
        }

    def test_a_wrong_weight_count_is_refused(self, criteria):
        with pytest.raises(McdaError, match="weights for 3 criteria"):
            critique_weights([0.5, 0.5], criteria, MATRIX)

    def test_negative_weights_are_refused(self, criteria):
        with pytest.raises(McdaError, match="non-negative"):
            critique_weights([-1.0, 1.0, 1.0], criteria, MATRIX)

    def test_all_zero_weights_are_refused(self, criteria):
        with pytest.raises(McdaError, match="non-negative"):
            critique_weights([0.0, 0.0, 0.0], criteria, MATRIX)


class TestNoDependency:
    def test_the_package_imports_no_llm_client(self):
        """The one-dependency claim is a differentiator, and an AI feature
        that quietly added a vendor SDK would forfeit it."""
        import pathlib

        import mcdakit

        root = pathlib.Path(mcdakit.__file__).parent
        forbidden = ("openai", "anthropic", "google.genai", "litellm", "requests")
        for path in root.rglob("*.py"):
            text = path.read_text()
            for name in forbidden:
                assert f"import {name}" not in text, f"{path} imports {name}"

    def test_any_callable_is_a_valid_provider(self):
        """No vendor is privileged; a lambda works."""
        proposal = propose_criteria("x", ask=lambda _p: GOOD_CRITERIA)
        assert len(proposal.criteria) == 2


class TestReadableOutput:
    """`print(proposal)` is how most callers will read the result, so its
    content is part of the contract rather than incidental."""

    def test_it_lists_criteria_with_direction_and_weight(self):
        text = str(propose_criteria("x", ask=replying(GOOD_CRITERIA)))
        assert "2 criteria proposed" in text
        assert "Price (cost, weight 0.50)" in text

    def test_it_shows_discarded_suggestions_and_the_reason(self):
        """A silently shortened list would misrepresent what the model said."""
        reply = """[
          {"name": "Price", "direction": "cost", "weight": 0.5},
          {"name": "Nonsense", "direction": "sideways", "weight": 0.5}
        ]"""
        text = str(propose_criteria("x", ask=replying(reply)))
        assert "discarded 'Nonsense'" in text
        assert "direction" in text, "the reason must appear, not just the name"

    def test_it_reports_agreement_when_measured(self):
        text = str(propose_criteria("x", ask=replying(GOOD_CRITERIA), samples=2))
        assert "agreement across replies: 100%" in text

    def test_it_omits_agreement_when_not_measured(self):
        """Printing a score for a check that never ran would be misleading."""
        assert "agreement" not in str(
            propose_criteria("x", ask=replying(GOOD_CRITERIA))
        )

    def test_it_states_that_nothing_was_applied(self):
        text = str(propose_criteria("x", ask=replying(GOOD_CRITERIA)))
        assert "not applied" in text
