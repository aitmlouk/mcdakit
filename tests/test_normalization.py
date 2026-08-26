"""Normalisation as a modelling choice, not a preprocessing detail.

The literature's point — and this package's — is that which scheme you pick
changes the numbers, sometimes the ranking, and how stable that ranking is.
These tests hold the schemes to their definitions and prove the choice is
consequential rather than cosmetic.
"""

import numpy as np
import pytest

from mcdakit import (
    Criterion,
    Decision,
    McdaError,
    available_normalizations,
    max_normalization,
    minmax_normalization,
    rank,
    register_normalization,
    reversal_check,
    sum_normalization,
    vector_normalization,
)
from mcdakit.normalization import NORMALIZATIONS, get_normalization, normalize


@pytest.fixture
def restore_schemes():
    saved = dict(NORMALIZATIONS)
    yield
    NORMALIZATIONS.clear()
    NORMALIZATIONS.update(saved)


DATA = np.array([[1.0, 10.0], [2.0, 20.0], [5.0, 40.0]])


class TestDefinitions:
    def test_vector_divides_by_the_euclidean_norm(self):
        out = vector_normalization(DATA)
        assert out[:, 0] == pytest.approx(DATA[:, 0] / np.sqrt((DATA[:, 0] ** 2).sum()))
        assert np.sum(out**2, axis=0) == pytest.approx([1.0, 1.0])

    def test_minmax_pins_the_range_to_zero_and_one(self):
        out = minmax_normalization(DATA)
        assert out.min(axis=0) == pytest.approx([0.0, 0.0])
        assert out.max(axis=0) == pytest.approx([1.0, 1.0])

    def test_max_pins_only_the_top(self):
        out = max_normalization(DATA)
        assert out.max(axis=0) == pytest.approx([1.0, 1.0])
        assert out.min(axis=0) == pytest.approx([0.2, 0.25])

    def test_sum_makes_each_column_a_share(self):
        out = sum_normalization(DATA)
        assert out.sum(axis=0) == pytest.approx([1.0, 1.0])

    @pytest.mark.parametrize("scheme", ["vector", "minmax", "max", "sum"])
    def test_shape_is_preserved(self, scheme):
        assert normalize(DATA, scheme).shape == DATA.shape


class TestDegenerateColumns:
    """A column that cannot discriminate must yield a constant, not a NaN.

    NaN sorts unpredictably, so one poisoned column would corrupt the ranking
    of every other criterion too.
    """

    @pytest.mark.parametrize("scheme", ["vector", "minmax", "max", "sum"])
    def test_a_flat_column_does_not_divide_by_zero(self, scheme):
        flat = np.array([[5.0, 1.0], [5.0, 9.0]])
        assert np.all(np.isfinite(normalize(flat, scheme)))

    @pytest.mark.parametrize("scheme", ["vector", "minmax", "max", "sum"])
    def test_a_zero_column_does_not_divide_by_zero(self, scheme):
        zeros = np.array([[0.0, 1.0], [0.0, 9.0]])
        assert np.all(np.isfinite(normalize(zeros, scheme)))

    @pytest.mark.parametrize("scheme", ["vector", "minmax", "max", "sum"])
    def test_identical_options_stay_finite(self, scheme):
        same = np.array([[3.0, 3.0], [3.0, 3.0]])
        assert np.all(np.isfinite(normalize(same, scheme)))


class TestItIsAModellingChoice:
    """The claim that justifies making this swappable at all."""

    CRITERIA = [
        Criterion("Price", 0.40, "cost"),
        Criterion("Quality", 0.35),
        Criterion("Support", 0.25),
    ]
    MATRIX = [[250.0, 8, 7], [200.0, 6, 8], [300.0, 9, 6], [210.0, 5, 5]]
    LABELS = ["A", "B", "C", "D"]

    def test_the_winner_can_change(self):
        """Same method, same data, different scheme, different answer."""
        vector = rank(
            self.MATRIX, self.CRITERIA, "topsis", self.LABELS, normalization="vector"
        )
        minmax = rank(
            self.MATRIX, self.CRITERIA, "topsis", self.LABELS, normalization="minmax"
        )
        assert vector.winner != minmax.winner, (
            "if the choice never mattered there would be nothing to expose"
        )

    def test_stability_can_change_too(self):
        """Rank reversal is a property of the normalisation as much as of the
        method — which is the whole reason the scheme is worth naming."""
        verdicts = {
            scheme: reversal_check(
                self.MATRIX, self.CRITERIA, "topsis", self.LABELS, normalization=scheme
            )["reversed"]
            for scheme in ("vector", "minmax", "max", "sum")
        }
        assert len(set(verdicts.values())) > 1, verdicts


class TestMethodDefaults:
    def test_each_method_declares_the_scheme_it_is_defined_with(self):
        from mcdakit.methods import Saw, Topsis, WeightedScoring

        assert Topsis.normalization == "vector"
        assert WeightedScoring.normalization == "minmax"
        assert Saw.normalization == "max"

    def test_the_default_reproduces_the_original_behaviour(self, supplier):
        """Refactoring normalisation out must not have moved any number."""
        result = rank(supplier, method="weighted_scoring")
        assert result.score_of("Supplier B") == pytest.approx(0.6, abs=1e-4)
        assert result.score_of("Supplier A") == pytest.approx(0.525, abs=1e-4)
        assert result.score_of("Supplier C") == pytest.approx(0.2833, abs=1e-4)

    def test_naming_the_default_explicitly_changes_nothing(self, supplier):
        assert rank(supplier, method="topsis").scores == pytest.approx(
            rank(supplier, method="topsis", normalization="vector").scores
        )

    def test_a_method_that_does_not_normalise_refuses_the_argument(self, supplier):
        """Accepting it silently would imply an effect that is not there."""
        with pytest.raises(McdaError, match="does not normalise"):
            rank(supplier, method="simple_scoring", normalization="minmax")

    def test_spotis_normalises_against_bounds_not_the_options(self, procurement):
        """SPOTIS's whole guarantee rests on its own fixed-bounds scaling, so
        it must not be swappable."""
        from mcdakit.methods import Spotis

        assert Spotis.normalization is None


class TestRegistration:
    def test_a_user_scheme_works_without_touching_the_library(self, restore_schemes):
        @register_normalization("test_double", summary="Twice min-max.")
        def _double(data):
            return 2.0 * minmax_normalization(data)

        assert "test_double" in available_normalizations()
        criteria = [Criterion("A", 1.0), Criterion("B", 1.0)]
        result = rank(
            [[1.0, 9.0], [9.0, 1.0]], criteria, "topsis", normalization="test_double"
        )
        assert np.all(np.isfinite(result.scores))

    def test_a_bare_callable_needs_no_registration(self):
        criteria = [Criterion("A", 1.0), Criterion("B", 1.0)]
        result = rank(
            [[1.0, 9.0], [9.0, 1.0]],
            criteria,
            "topsis",
            normalization=minmax_normalization,
        )
        assert np.all(np.isfinite(result.scores))

    def test_a_duplicate_name_is_refused(self, restore_schemes):
        with pytest.raises(McdaError, match="already registered"):
            register_normalization("minmax", minmax_normalization)

    def test_a_duplicate_can_be_replaced_deliberately(self, restore_schemes):
        register_normalization("minmax", minmax_normalization, replace=True)
        assert "minmax" in available_normalizations()

    def test_an_unknown_name_lists_the_real_ones(self):
        with pytest.raises(McdaError, match="Unknown normalization"):
            get_normalization("nonsense")

    def test_a_non_callable_is_refused(self, restore_schemes):
        with pytest.raises(McdaError, match="must be callable"):
            register_normalization("test_bad", "not a function")


class TestSchemeOutputIsValidated:
    def test_a_scheme_changing_the_shape_is_caught(self):
        def _shrink(data):
            return data[:1]

        with pytest.raises(McdaError, match="changed the matrix shape"):
            normalize(DATA, _shrink)

    def test_a_scheme_producing_nan_is_caught(self):
        """Left unchecked this would surface much later as an unrankable
        result, with nothing pointing back at the normalisation."""

        def _divide_by_zero(data):
            return data / np.zeros_like(data)

        with (
            np.errstate(invalid="ignore", divide="ignore"),
            pytest.raises(McdaError, match="non-finite"),
        ):
            normalize(DATA, _divide_by_zero)

    def test_a_non_callable_scheme_is_refused(self):
        with pytest.raises(McdaError, match="name or a callable"):
            normalize(DATA, 42)


class TestWeightNormalisation:
    """The guard every method used to carry its own copy of.

    Unreachable through `rank()` — a Decision refuses all-zero weights — but
    the algorithm functions are public and callable on hand-built arrays, so
    the branch is real and needs testing rather than deleting.
    """

    def test_weights_are_scaled_to_sum_to_one(self):
        from mcdakit.normalization import normalize_weights

        assert normalize_weights(np.array([2.0, 3.0, 5.0])) == pytest.approx(
            [0.2, 0.3, 0.5]
        )

    def test_scale_does_not_matter(self):
        from mcdakit.normalization import normalize_weights

        assert normalize_weights(np.array([1.0, 3.0])) == pytest.approx(
            normalize_weights(np.array([100.0, 300.0]))
        )

    def test_all_zero_weights_return_none(self):
        """None rather than a fallback: inventing preferences nobody
        expressed would be worse than saying nothing can be ranked."""
        from mcdakit.normalization import normalize_weights

        assert normalize_weights(np.array([0.0, 0.0])) is None

    @pytest.mark.parametrize(
        "function",
        ["simple_scoring", "weighted_scoring", "saw", "topsis", "vikor", "electre"],
    )
    def test_each_method_returns_flat_zeros_on_zero_weights(self, function):
        """Called directly, with weights rank() would have refused."""
        import mcdakit.methods as m

        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        zero = np.array([0.0, 0.0])
        scores = getattr(m, function)(data, zero)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
        if function != "simple_scoring":
            assert np.all(scores == 0.0)

    def test_spotis_returns_zeros_and_keeps_its_guarantee(self):
        from mcdakit.methods import spotis

        scores, reversal_free, messages = spotis(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([0.0, 0.0]),
            ["benefit", "benefit"],
            [(0.0, 10.0), (0.0, 10.0)],
        )
        assert np.all(scores == 0.0)
        assert reversal_free is True
        assert messages == ()


class TestContextNormalize:
    """`ctx.normalize()` is what docs/extending.md tells contributors to use."""

    def test_the_documented_pattern_works(self):
        from mcdakit import Method, register
        from mcdakit.methods.registry import unregister

        class Documented(Method):
            name = "test_ctx_normalize"
            summary = "Uses ctx.normalize(), as the guide instructs."
            citation = "n/a"
            normalization = "minmax"

            def score(self, ctx):
                return ctx.normalize() @ ctx.normalized_weights

        register(Documented())
        try:
            criteria = [Criterion("Price", 0.5, "cost"), Criterion("Q", 0.5)]
            result = rank(
                [[250.0, 8], [200.0, 6]], criteria, "test_ctx_normalize", ["A", "B"]
            )
            assert np.all(np.isfinite(result.scores))

            overridden = rank(
                [[250.0, 8], [200.0, 6]],
                criteria,
                "test_ctx_normalize",
                ["A", "B"],
                normalization="max",
            )
            assert not np.allclose(result.scores, overridden.scores), (
                "the caller's override must reach ctx.normalize()"
            )
        finally:
            unregister("test_ctx_normalize")

    def test_a_method_without_a_scheme_gets_its_data_back(self):
        from mcdakit.methods.base import ScoringContext

        decision = Decision([[1.0, 2.0], [3.0, 4.0]], [Criterion("a"), Criterion("b")])
        ctx = ScoringContext(
            data=np.asarray(decision.matrix),
            weights=decision.weights,
            decision=decision,
        )
        assert np.allclose(ctx.normalize(), decision.matrix)

    def test_it_accepts_an_explicit_matrix(self):
        from mcdakit.methods.base import ScoringContext

        decision = Decision([[1.0, 2.0], [3.0, 4.0]], [Criterion("a"), Criterion("b")])
        ctx = ScoringContext(
            data=np.asarray(decision.matrix),
            weights=decision.weights,
            decision=decision,
            _normalization="max",
        )
        other = np.array([[2.0], [4.0]])
        assert ctx.normalize(other).ravel() == pytest.approx([0.5, 1.0])
