"""The contributor conformance suite — the check a method author runs.

Its job is to catch the quiet failures: an inverted score convention, a NaN on
a degenerate input, direction handled twice. Each test here is a method that
is wrong in one specific way, asserting the suite notices.
"""

import numpy as np
import pytest

from mcdakit import McdaError, Method, Wants
from mcdakit.methods import BUILTIN_METHODS
from mcdakit.testing import (
    CHECKS,
    ConformanceError,
    check_method,
    conformance_report,
)
from mcdakit.testing import _main as run_cli


class Sound(Method):
    """A well-behaved method, used as the control."""

    name = "test_sound"
    summary = "Weighted total over the oriented matrix."
    citation = "n/a — test fixture"

    def score(self, ctx):
        return ctx.data @ ctx.normalized_weights


class TestTheControl:
    def test_a_sound_method_passes_everything(self):
        check_method(Sound())

    def test_the_report_lists_every_check(self):
        report = conformance_report(Sound())
        assert len(report) == len(CHECKS)
        assert all(why is None for why in report.values())


class TestBuiltins:
    """Every shipped method must pass the suite we ask contributors to pass.

    Holding contributors to a standard the library itself fails would be
    indefensible, and this is what stops that happening quietly.
    """

    @pytest.mark.parametrize("cls", BUILTIN_METHODS, ids=lambda c: c.name)
    def test_it_conforms(self, cls):
        method = cls()
        report = conformance_report(method, strict=False)
        failures = {k: v for k, v in report.items() if v}
        assert not failures, failures


class TestCatchesInvertedScores:
    def test_a_smallest_is_best_measure_is_caught(self):
        """The failure that produces a confident, exactly inverted ranking."""

        class Distance(Method):
            name = "test_distance"
            summary = "Distance from the ideal — forgot to negate."
            citation = "n/a"

            def score(self, ctx):
                return np.linalg.norm(ctx.data - ctx.data.max(axis=0), axis=1)

        report = conformance_report(Distance())
        assert report["higher is better"]
        assert "negate" in report["higher is better"]


class TestCatchesDirectionMistakes:
    def test_reading_directions_while_oriented_is_caught(self):
        """The mistake that silently reverses cost criteria: the library has
        already flipped them, and the method flips them back."""

        class DoubleHandled(Method):
            name = "test_double"
            summary = "Handles direction itself but forgot Wants.RAW."
            citation = "n/a"

            def score(self, ctx):
                is_cost = np.array([d == "cost" for d in ctx.directions])
                data = ctx.data.copy()
                data[:, is_cost] = -data[:, is_cost]
                return data @ ctx.normalized_weights

        report = conformance_report(DoubleHandled())
        assert report["scores every option"]
        assert "Wants.RAW" in report["scores every option"]

    def test_a_cost_criterion_ranking_backwards_is_caught(self):
        class Backwards(Method):
            name = "test_backwards"
            summary = "Ignores orientation entirely."
            citation = "n/a"
            wants = Wants.RAW

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        report = conformance_report(Backwards())
        assert report["handles cost criteria"]

    def test_a_raw_method_that_gets_oriented_data_is_caught(self):
        """Guards the plumbing itself, not just the method."""
        report = conformance_report(Sound())
        assert report["receives the data form it asked for"] is None


class TestCatchesNumericalProblems:
    def test_a_nan_on_a_flat_criterion_is_caught(self):
        class Fragile(Method):
            name = "test_fragile"
            summary = "Divides by the observed range."
            citation = "n/a"

            def score(self, ctx):
                span = ctx.data.max(axis=0) - ctx.data.min(axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    # The division by zero is the point of the fixture; numpy
                    # need not also warn about it.
                    normalised = (ctx.data - ctx.data.min(axis=0)) / span
                return normalised @ ctx.normalized_weights

        report = conformance_report(Fragile())
        assert report["scores stay finite"]

    def test_a_wrong_shaped_return_is_caught(self):
        class WrongShape(Method):
            name = "test_wrong_shape"
            summary = "Returns one score whatever the problem."
            citation = "n/a"

            def score(self, ctx):
                return np.array([1.0])

        assert conformance_report(WrongShape())["scores every option"]


class TestCatchesInvariantBreaks:
    def test_weight_scale_dependence_is_caught(self):
        class ScaleDependent(Method):
            name = "test_scale"
            summary = "Uses raw weights where it needs normalised ones."
            citation = "n/a"

            def score(self, ctx):
                # Order flips with the weight magnitude: the penalty term
                # grows quadratically while the reward grows linearly.
                total = ctx.data @ ctx.weights
                return total - 0.01 * total**2

        report = conformance_report(ScaleDependent())
        assert report["weight scale does not matter"]

    def test_row_order_dependence_is_caught(self):
        class OrderDependent(Method):
            name = "test_order"
            summary = "Rewards being listed first."
            citation = "n/a"

            def score(self, ctx):
                bonus = np.linspace(1.0, 0.0, ctx.n_options)
                return ctx.data @ ctx.normalized_weights + bonus

        report = conformance_report(OrderDependent())
        assert report["option order does not matter"]


class TestReversalFreedomClaim:
    def test_a_false_claim_is_caught(self):
        """`reversal_free` is the strongest claim the library makes, so a
        method asserting it without holding it must not get away with it."""

        class Liar(Method):
            name = "test_liar"
            summary = "Min-max normalised, but claims reversal-freedom."
            citation = "n/a"
            reversal_free = True

            def score(self, ctx):
                lo = ctx.data.min(axis=0)
                span = np.where(
                    ctx.data.max(axis=0) - lo == 0, 1.0, ctx.data.max(axis=0) - lo
                )
                return ((ctx.data - lo) / span) @ ctx.normalized_weights

        report = conformance_report(Liar())
        assert report["reversal-freedom holds if claimed"]

    def test_the_claim_is_not_checked_when_not_made(self):
        assert conformance_report(Sound())["reversal-freedom holds if claimed"] is None

    def test_spotis_makes_the_claim_and_keeps_it(self):
        from mcdakit.methods import Spotis

        assert Spotis().reversal_free is True
        assert conformance_report(Spotis())["reversal-freedom holds if claimed"] is None


class TestMetadata:
    def test_missing_metadata_fails_under_strict(self):
        class Bare(Method):
            name = "test_bare"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        report = conformance_report(Bare(), strict=True)
        assert report["describes itself"]
        assert report["cites its source"]

    def test_missing_metadata_is_tolerated_when_prototyping(self):
        class Bare(Method):
            name = "test_bare_2"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        check_method(Bare(), strict=False)


class TestErrorReporting:
    def test_every_failure_is_listed_not_just_the_first(self):
        class Bad(Method):
            name = "test_bad"

            def score(self, ctx):
                return -(ctx.data @ ctx.normalized_weights)

        with pytest.raises(ConformanceError) as caught:
            check_method(Bad())
        message = str(caught.value)
        assert message.count("  - ") >= 3
        assert "conformance checks" in message

    def test_the_method_is_named_by_its_own_name(self):
        class Named(Method):
            name = "test_named_method"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return -(ctx.data @ ctx.normalized_weights)

        report = conformance_report(Named())
        assert "__conformance" not in str(report)

    def test_it_refuses_something_that_is_not_a_method(self):
        from mcdakit import McdaError

        with pytest.raises(McdaError, match="takes a Method instance"):
            conformance_report(object())


class TestRegistryIsLeftAlone:
    def test_checking_does_not_leave_the_method_registered(self):
        from mcdakit.methods.registry import has

        check_method(Sound())
        assert not has("test_sound")

    def test_the_methods_name_is_restored(self):
        method = Sound()
        conformance_report(method)
        assert method.name == "test_sound"

    def test_an_already_registered_method_is_not_displaced(self):
        from mcdakit.methods.registry import get

        original = get("topsis")
        conformance_report(original)
        assert get("topsis") is original


class TestCommandLine:
    def test_it_reports_success(self, capsys):
        code = run_cli(["mcdakit.methods:Topsis"])
        assert code == 0
        assert "All" in capsys.readouterr().out

    def test_it_reports_failure_with_a_non_zero_exit(self, capsys):
        code = run_cli(["tests.test_testing:_CliFailure"])
        assert code == 1
        assert "FAIL" in capsys.readouterr().out

    def test_a_malformed_target_is_refused(self):
        with pytest.raises(SystemExit):
            run_cli(["not_a_valid_target"])


class _CliFailure(Method):
    """Deliberately wrong, for the command-line test above."""

    name = "test_cli_failure"
    summary = "x"
    citation = "y"

    def score(self, ctx):
        return -(ctx.data @ ctx.normalized_weights)


class TestFailureMessagesAreExercised:
    """The suite's own error text.

    Untested error messages are how a check ships saying the wrong thing — and
    these are the messages a contributor reads when their method is broken, so
    they are the ones that most need to be right.
    """

    def test_a_nameless_method_is_reported(self):
        class Nameless(Method):
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        report = conformance_report(Nameless())
        assert "`name` is empty" in report["declares a name"]

    def test_a_wrong_score_count_is_reported(self):
        class TooFew(Method):
            name = "test_too_few"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return np.ones(2)

        report = conformance_report(TooFew())
        assert "for 3 options" in report["scores every option"]

    def test_a_non_finite_score_names_the_input_that_caused_it(self):
        class Nan(Method):
            name = "test_always_nan"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return np.full(ctx.n_options, np.nan)

        report = conformance_report(Nan())
        assert "non-finite score on" in report["scores stay finite"]

    def test_failing_on_a_single_option_is_reported(self):
        class NeedsTwo(Method):
            name = "test_needs_two"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                if ctx.n_options < 2:
                    raise ValueError("I need at least two")
                return ctx.data @ ctx.normalized_weights

        report = conformance_report(NeedsTwo())
        assert report["handles a single option"]

    def test_the_data_form_check_passes_for_both_declarations(self):
        """It reports nothing when the plumbing is sound, which is the only
        state reachable from outside.

        The mismatch messages themselves are a guard against `rank()` handing
        a method the matrix it did not ask for. Reaching them would mean
        breaking orientation first, so they stay uncovered by design rather
        than by neglect — see the pragma in mcdakit/testing.py.
        """
        from mcdakit.testing import _check_raw_methods_see_raw_data

        class Raw(Method):
            name = "test_form_raw"
            summary = "x"
            citation = "y"
            wants = Wants.RAW

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        class Oriented(Method):
            name = "test_form_oriented"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        assert _check_raw_methods_see_raw_data(Raw()) is None
        assert _check_raw_methods_see_raw_data(Oriented()) is None

    def test_the_registry_is_restored_when_a_name_collides(self):
        """The suite registers under the method's own name where it can. If
        that name is taken, the original must come back afterwards."""
        from mcdakit.methods.registry import get

        class Shadow(Method):
            name = "topsis"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        original = get("topsis")
        conformance_report(Shadow())
        assert get("topsis") is original


class TestChecksCalledDirectly:
    """Some messages are only returned when a check is driven on its own.

    Through `conformance_report` an earlier failing check often raises first,
    so these lines are reached but the message itself never is. Driving each
    check directly exercises the text a contributor actually reads.

    Two messages — the score-count and single-option ones — are deliberately
    absent: `Result` validates shape and finiteness before a check can look,
    so those returns are redundant belt-and-braces rather than live paths.
    """

    def _method(self, name, body, **attrs):
        namespace = {
            "name": name,
            "summary": "x",
            "citation": "y",
            "score": body,
            **attrs,
        }
        return type("Probe", (Method,), namespace)()

    def test_a_non_finite_score_message_names_the_case(self):
        from mcdakit.testing import _check_scores_are_finite

        method = self._method(
            "test_direct_nan",
            lambda self, ctx: np.full(ctx.n_options, np.inf),
        )
        message = _check_scores_are_finite(method) or ""
        assert "non-finite score on" in message
        assert "criterion every option scores identically" in message

    def test_a_method_that_never_scores_is_tolerated_by_the_form_check(self):
        """If `score` was never reached there is nothing to judge, so the
        check reports nothing rather than guessing."""
        from mcdakit.testing import _check_raw_methods_see_raw_data

        class Refuses(Method):
            name = "test_direct_refuses"
            summary = "x"
            citation = "y"

            def validate(self, decision):
                raise McdaError("not for me")

            def score(self, ctx):  # pragma: no cover - never reached
                return np.zeros(ctx.n_options)

        with pytest.raises(McdaError):
            _check_raw_methods_see_raw_data(Refuses())

    def test_an_instance_level_score_is_restored(self):
        """The form check swaps `score` for a spy. A method carrying its own
        instance attribute must get it back, not lose it to `del`."""
        from mcdakit.testing import _check_raw_methods_see_raw_data

        class Plain(Method):
            name = "test_direct_restore"
            summary = "x"
            citation = "y"

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        method = Plain()
        own = lambda ctx: np.zeros(ctx.n_options)  # noqa: E731
        method.score = own
        _check_raw_methods_see_raw_data(method)
        assert method.score is own, "an instance-level score must survive"


class TestDeclaredDomain:
    """`requires_positive` exempts a method from the finiteness check, so the
    suite verifies the exemption is earned rather than merely claimed."""

    def test_a_method_claiming_the_exemption_without_earning_it_is_caught(self):
        class Pretender(Method):
            name = "test_pretender"
            summary = "Claims a positive-only domain but accepts anything."
            citation = "n/a"
            requires_positive = True

            def score(self, ctx):
                return ctx.data @ ctx.normalized_weights

        report = conformance_report(Pretender())
        assert report["a declared domain is enforced"]
        assert (
            "refuse input outside its domain"
            in (report["a declared domain is enforced"])
        )

    def test_a_method_that_enforces_it_passes(self):
        from mcdakit.methods import Waspas

        assert conformance_report(Waspas())["a declared domain is enforced"] is None

    def test_the_check_is_skipped_for_methods_without_the_declaration(self):
        from mcdakit.methods import Topsis

        assert conformance_report(Topsis())["a declared domain is enforced"] is None
