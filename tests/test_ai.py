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


class TestOptIn:
    """Using a model is a choice the source code should record.

    MCDA results are often used to justify a decision to somebody else.
    Whether a language model contributed to the inputs is part of how that
    result was reached, so reaching for it requires an explicit import rather
    than being available by default on the package namespace.
    """

    def test_importing_mcdakit_does_not_load_the_ai_module(self):
        import subprocess
        import sys

        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, mcdakit; print('mcdakit.ai' in sys.modules)",
            ],
            capture_output=True,
            text=True,
        )
        assert out.stdout.strip() == "False", out.stdout + out.stderr

    def test_the_ai_surface_is_not_on_the_package_namespace(self):
        import mcdakit

        for name in ("propose_criteria", "propose_weights", "critique_weights"):
            assert not hasattr(mcdakit, name), (
                f"{name} should require `from mcdakit.ai import ...`"
            )
            assert name not in mcdakit.__all__

    def test_the_explicit_import_works(self):
        from mcdakit.ai import propose_criteria as imported

        assert callable(imported)

    def test_the_module_needs_no_extra_dependency_to_import(self):
        """Opt-in by import, not by installation: there is no `pip install
        mcdakit[ai]` to forget, and no vendor SDK to acquire."""
        import ast
        import pathlib

        import mcdakit

        source = (pathlib.Path(mcdakit.__file__).parent / "ai.py").read_text()
        top = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                top |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top.add((node.module or "").split(".")[0])
        assert top <= {
            "__future__",
            "collections",
            "dataclasses",
            "json",
            "numpy",
            "re",
            "typing",
        }, f"unexpected imports: {top}"


CONSISTENT_PAIRS = (
    '[{"i":0,"j":1,"ratio":3},{"i":0,"j":2,"ratio":9},{"i":1,"j":2,"ratio":3}]'
)


class TestProposeComparisons:
    """Pairwise judgements are worth asking for because they are checkable.

    A weight vector cannot contradict itself; a set of pairwise ratios can,
    and Saaty's consistency ratio detects it without anyone knowing what the
    right weights were.
    """

    def test_consistent_judgements_give_the_expected_weights(self, criteria):
        from mcdakit.ai import propose_comparisons

        proposal = propose_comparisons("x", criteria, ask=replying(CONSISTENT_PAIRS))
        assert proposal.consistency_ratio == pytest.approx(0.0, abs=1e-9)
        assert proposal.consistent
        # 3:1 and 9:1 imply priorities in 9:3:1.
        assert proposal.weights == pytest.approx([9 / 13, 3 / 13, 1 / 13], abs=1e-4)

    def test_a_self_contradicting_model_is_caught(self, criteria):
        """A over B, B over C, but C over A cannot all hold, and the model
        gets no say in whether that is noticed."""
        from mcdakit.ai import propose_comparisons

        circular = (
            '[{"i":0,"j":1,"ratio":5},{"i":1,"j":2,"ratio":5},'
            '{"i":0,"j":2,"ratio":0.2}]'
        )
        proposal = propose_comparisons("x", criteria, ask=replying(circular))
        assert proposal.consistency_ratio > 0.10
        assert not proposal.consistent

    def test_inconsistency_is_reported_not_raised(self, criteria):
        """The analyst decides whether to proceed, as elsewhere in AHP."""
        from mcdakit.ai import propose_comparisons

        circular = (
            '[{"i":0,"j":1,"ratio":9},{"i":1,"j":2,"ratio":9},'
            '{"i":0,"j":2,"ratio":0.111}]'
        )
        proposal = propose_comparisons("x", criteria, ask=replying(circular))
        assert proposal.weights.sum() == pytest.approx(1.0)

    def test_the_printed_output_flags_a_contradiction(self, criteria):
        from mcdakit.ai import propose_comparisons

        circular = (
            '[{"i":0,"j":1,"ratio":5},{"i":1,"j":2,"ratio":5},'
            '{"i":0,"j":2,"ratio":0.2}]'
        )
        text = str(propose_comparisons("x", criteria, ask=replying(circular)))
        assert "INCONSISTENT" in text
        assert "contradict" in text

    def test_a_malformed_comparison_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="malformed"):
            propose_comparisons("x", criteria, ask=replying('[{"i":0}]'))

    def test_an_out_of_range_pair_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="out of range"):
            propose_comparisons(
                "x", criteria, ask=replying('[{"i":0,"j":9,"ratio":3}]')
            )

    def test_comparing_a_criterion_with_itself_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="with itself"):
            propose_comparisons(
                "x", criteria, ask=replying('[{"i":1,"j":1,"ratio":3}]')
            )

    def test_a_non_positive_ratio_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="positive finite ratio"):
            propose_comparisons(
                "x", criteria, ask=replying('[{"i":0,"j":1,"ratio":0}]')
            )

    def test_an_empty_reply_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="no comparisons"):
            propose_comparisons("x", criteria, ask=replying("[]"))

    def test_a_json_object_where_a_list_was_required_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(AiError, match="Expected a JSON list"):
            propose_comparisons("x", criteria, ask=replying('{"i":0}'))

    def test_a_single_criterion_is_refused(self):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(McdaError, match="at least two criteria"):
            propose_comparisons(
                "x", [Criterion("Only", 1.0)], ask=replying(CONSISTENT_PAIRS)
            )

    def test_zero_samples_is_refused(self, criteria):
        from mcdakit.ai import propose_comparisons

        with pytest.raises(McdaError, match="at least 1"):
            propose_comparisons(
                "x", criteria, ask=replying(CONSISTENT_PAIRS), samples=0
            )

    def test_repeated_asking_measures_self_consistency(self, criteria):
        from mcdakit.ai import propose_comparisons

        other = (
            '[{"i":0,"j":1,"ratio":1},{"i":0,"j":2,"ratio":1},{"i":1,"j":2,"ratio":1}]'
        )
        same = propose_comparisons(
            "x", criteria, ask=replying(CONSISTENT_PAIRS), samples=2
        )
        differing = propose_comparisons(
            "x", criteria, ask=replying(CONSISTENT_PAIRS, other), samples=2
        )
        assert same.agreement == pytest.approx(1.0)
        assert differing.agreement < same.agreement


PANEL_VIEWS = {
    "cost analyst": "[[2.75,7,8],[2.90,8.5,8],[3.40,9,7]]",
    "quality lead": "[[2.75,9,8],[2.90,6.0,8],[3.40,9,7]]",
    "ops manager": "[[2.75,7,9],[2.90,8.5,6],[3.40,9,9]]",
}


def panel_model(prompt):
    for persona, reply in PANEL_VIEWS.items():
        if persona in prompt:
            return reply
    return "no json here"


class TestSimulatePanel:
    """A simulated panel is reproducible where a human one is not, at the
    price of being a model's impression of an expert. That trade is only
    acceptable if the output goes through the same scrutiny."""

    LABELS = ["A", "B", "C"]

    def test_each_persona_becomes_a_participant(self, criteria):
        from mcdakit.ai import simulate_panel

        panel = simulate_panel(
            "x", criteria, list(PANEL_VIEWS), self.LABELS, ask=panel_model
        )
        assert len(panel.participants) == 3
        assert {p.name for p in panel.participants} == set(PANEL_VIEWS)

    def test_the_panel_feeds_the_existing_group_machinery(self, criteria):
        """The point of returning Participant objects: a simulated panel is
        scrutinised exactly as a real one is."""
        from mcdakit.ai import simulate_panel
        from mcdakit.group import group_rank

        panel = simulate_panel(
            "x", criteria, list(PANEL_VIEWS), self.LABELS, ask=panel_model
        )
        out = group_rank(panel.participants, criteria, labels=self.LABELS)
        assert out["result"].winner in self.LABELS
        assert "most_contested_option" in out["disagreement"]

    def test_spread_measures_whether_the_personas_actually_differed(self, criteria):
        """A panel whose members agree exactly added nothing over asking once,
        and saying so is more useful than reporting a false consensus."""
        from mcdakit.ai import simulate_panel

        varied = simulate_panel(
            "x", criteria, list(PANEL_VIEWS), self.LABELS, ask=panel_model
        )
        identical = simulate_panel(
            "x",
            criteria,
            ["one", "two"],
            self.LABELS,
            ask=replying("[[1,2,3],[4,5,6],[7,8,9]]"),
        )
        assert varied.spread > 0
        assert identical.spread == pytest.approx(0.0)
        assert "adds nothing" in str(identical)

    def test_a_malformed_persona_is_discarded_with_its_reason(self, criteria):
        from mcdakit.ai import simulate_panel

        personas = [*PANEL_VIEWS, "broken"]
        panel = simulate_panel("x", criteria, personas, self.LABELS, ask=panel_model)
        assert len(panel.participants) == 3
        assert panel.rejected and panel.rejected[0][0] == "broken"
        assert "discarded 'broken'" in str(panel)

    def test_a_wrong_shaped_reply_is_discarded(self, criteria):
        from mcdakit.ai import simulate_panel

        def wrong(prompt):
            return "[[1,2,3]]" if "narrow" in prompt else PANEL_VIEWS["cost analyst"]

        panel = simulate_panel(
            "x",
            criteria,
            ["cost analyst", "quality lead", "narrow"],
            self.LABELS,
            ask=wrong,
        )
        assert any("3x3" in why or "matrix" in why for _p, why in panel.rejected)

    def test_a_reply_containing_nan_is_discarded(self, criteria):
        from mcdakit.ai import simulate_panel

        def nan_reply(prompt):
            if "bad" in prompt:
                return '[[1,2,3],[4,"NaN",6],[7,8,9]]'
            return PANEL_VIEWS["cost analyst"]

        panel = simulate_panel(
            "x",
            criteria,
            ["cost analyst", "quality lead", "bad"],
            self.LABELS,
            ask=nan_reply,
        )
        assert any("NaN" in why or "infinity" in why for _p, why in panel.rejected)

    def test_fewer_than_two_usable_personas_is_refused(self, criteria):
        from mcdakit.ai import simulate_panel

        with pytest.raises(AiError, match="Fewer than two personas"):
            simulate_panel(
                "x", criteria, ["a", "b"], self.LABELS, ask=replying("nonsense")
            )

    def test_a_panel_of_one_is_refused(self, criteria):
        from mcdakit.ai import simulate_panel

        with pytest.raises(McdaError, match="at least two personas"):
            simulate_panel("x", criteria, ["only"], self.LABELS, ask=panel_model)

    def test_nothing_is_applied_automatically(self, criteria):
        from mcdakit.ai import simulate_panel

        panel = simulate_panel(
            "x", criteria, list(PANEL_VIEWS), self.LABELS, ask=panel_model
        )
        assert panel.accepted is False
        assert "not applied" in str(panel)


class TestNarrate:
    """The model is given computed facts and asked to write them up, so the
    only failure available to it is misstating a number — which is checkable.
    """

    def _result(self, criteria):
        from mcdakit import Decision, rank

        return rank(
            Decision(
                [[2.75, 7, 8], [2.90, 8.5, 8], [3.40, 9, 7]], criteria, ["A", "B", "C"]
            ),
            method="weighted_scoring",
        )

    def test_a_faithful_paragraph_passes(self, criteria):
        from mcdakit.ai import narrate

        result = self._result(criteria)
        top = f"{result.ranking[0][1]:.4f}"
        out = narrate(
            result, ask=replying(f"{result.winner} is recommended, scoring {top}.")
        )
        assert out["faithful"]
        assert out["unsupported_numbers"] == ()

    def test_an_invented_figure_is_caught(self, criteria):
        """The failure this check exists for: a confident number the model
        made up, pasted into a decision report."""
        from mcdakit.ai import narrate

        out = narrate(
            self._result(criteria),
            ask=replying("A is recommended with 97.3% confidence."),
        )
        assert not out["faithful"]
        assert "97.3" in out["unsupported_numbers"]

    def test_the_sensitivity_figures_are_offered_and_accepted(self, criteria):
        from mcdakit import sensitivity
        from mcdakit.ai import narrate

        result = self._result(criteria)
        report = sensitivity(result)
        out = narrate(result, ask=replying("Stable enough."), sensitivity_report=report)
        assert report["level"] in out["facts"]

    def test_the_facts_given_to_the_model_are_returned(self, criteria):
        from mcdakit.ai import narrate

        out = narrate(self._result(criteria), ask=replying("Anything."))
        assert "Method: weighted_scoring" in out["facts"]
        assert "Recommended:" in out["facts"]

    def test_it_takes_a_result_not_a_decision(self, criteria):
        from mcdakit import Decision
        from mcdakit.ai import narrate

        decision = Decision([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], criteria, ["A", "B"])
        with pytest.raises(McdaError, match="takes a Result"):
            narrate(decision, ask=replying("x"))


class TestUncoveredPaths:
    """Cases the happy path does not reach, each a real possibility."""

    def test_a_consistent_proposal_prints_no_contradiction_warning(self, criteria):
        from mcdakit.ai import propose_comparisons

        text = str(propose_comparisons("x", criteria, ask=replying(CONSISTENT_PAIRS)))
        assert "contradict" not in text
        assert "INCONSISTENT" not in text

    def test_replies_sharing_no_pair_yield_no_agreement_score(self, criteria):
        """Two replies that compare different pairs cannot be compared, and
        inventing a similarity for them would be worse than reporting none."""
        from mcdakit.ai import propose_comparisons

        first = '[{"i":0,"j":1,"ratio":3}]'
        second = '[{"i":1,"j":2,"ratio":3}]'
        proposal = propose_comparisons(
            "x", criteria, ask=replying(first, second), samples=2
        )
        assert proposal.agreement is None

    def test_an_immovable_result_narrates_without_a_tolerance(self, criteria):
        """sensitivity() reports overall=None when no weight change unseats
        the winner; the narration must omit the figure rather than print
        None."""
        from mcdakit import Criterion, Decision, rank, sensitivity
        from mcdakit.ai import narrate

        dominant = [
            Criterion("A", 1.0, "benefit"),
            Criterion("B", 1.0, "benefit"),
        ]
        result = rank(
            Decision([[9.0, 9.0], [1.0, 1.0]], dominant, ["Strong", "Weak"]),
            method="weighted_scoring",
        )
        report = sensitivity(result)
        assert report["overall"] is None, "fixture must be immovable"

        out = narrate(
            result, ask=replying("Strong is recommended."), sensitivity_report=report
        )
        assert "Stability: immovable" in out["facts"]
        assert "moves by" not in out["facts"]
        assert out["faithful"]
