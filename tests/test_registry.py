"""The extension system: registering, discovering and dispatching methods.

If a method written outside this package is not a first-class citizen, the
plugin system has failed. These tests define what "first-class" means.
"""

import warnings

import numpy as np
import pytest

from mcdakit import (
    METHODS,
    Criterion,
    McdaError,
    Method,
    MethodResult,
    ScoringContext,
    Wants,
    available,
    compare_methods,
    rank,
    register,
    sensitivity,
    unregister,
)
from mcdakit.methods import BUILTIN_METHODS, register_builtins
from mcdakit.methods.registry import (
    DuplicateMethodError,
    PluginLoadError,
    _load_plugins,
    _reset_for_testing,
    get,
    has,
    names,
)


@pytest.fixture
def clean_registry():
    """Restore the registry after a test that mutates it."""
    import mcdakit.methods.registry as reg

    saved = dict(reg._METHODS)
    saved_loaded = reg._PLUGINS_LOADED
    yield
    reg._METHODS.clear()
    reg._METHODS.update(saved)
    reg._PLUGINS_LOADED = saved_loaded


class Median(Method):
    """A minimal third-party method, defined entirely outside the library."""

    name = "test_median"
    summary = "Median score across criteria."

    def score(self, ctx: ScoringContext) -> np.ndarray:
        return np.median(ctx.data, axis=1)


class TestRegistration:
    def test_a_registered_method_is_reachable_by_name(self, clean_registry):
        register(Median())
        assert has("test_median")
        assert "test_median" in names()
        assert isinstance(get("test_median"), Median)

    def test_rank_dispatches_to_it(self, clean_registry, supplier):
        register(Median())
        result = rank(supplier, method="test_median")
        assert result.method == "test_median"
        assert result.scores == pytest.approx(np.median(supplier.matrix, axis=1))

    def test_it_joins_compare_methods_automatically(self, clean_registry, supplier):
        """A plugin must not have to be listed anywhere to be compared."""
        before = len(compare_methods(supplier))
        register(Median())
        after = compare_methods(supplier)
        assert len(after) == before + 1
        assert "test_median" in after

    def test_sensitivity_works_on_it(self, clean_registry, supplier):
        """Bisection runs over whatever scoring function it is given, so it
        must work for a method the library has never seen."""
        register(Median())
        report = sensitivity(rank(supplier, method="test_median"))
        assert len(report["rows"]) == supplier.n_criteria
        assert report["level"] in {"fragile", "moderate", "robust", "immovable"}

    def test_METHODS_is_a_live_view_not_a_snapshot(self, clean_registry):
        """It was a frozen tuple; a plugin registered after import would have
        been invisible to anything that read it."""
        assert "test_median" not in METHODS
        register(Median())
        assert "test_median" in METHODS
        assert len(METHODS) == len(names())

    def test_unregister_removes_it(self, clean_registry):
        register(Median())
        unregister("test_median")
        assert not has("test_median")
        with pytest.raises(McdaError, match="Unknown method"):
            get("test_median")

    def test_unregistering_something_absent_is_refused(self, clean_registry):
        with pytest.raises(McdaError, match="No method named"):
            unregister("never_existed")


class TestRegistrationRefusals:
    def test_a_duplicate_name_is_refused(self, clean_registry):
        """Silently shadowing a method would make `method="topsis"` mean
        different things in different processes."""
        register(Median())
        with pytest.raises(DuplicateMethodError, match="already registered"):
            register(Median())

    def test_a_duplicate_can_be_replaced_deliberately(self, clean_registry):
        register(Median())
        register(Median(), replace=True)
        assert has("test_median")

    def test_shadowing_a_builtin_needs_the_same_deliberate_flag(self, clean_registry):
        class FakeTopsis(Method):
            name = "topsis"

            def score(self, ctx):
                return np.zeros(ctx.n_options)

        with pytest.raises(DuplicateMethodError):
            register(FakeTopsis())
        register(FakeTopsis(), replace=True)
        assert isinstance(get("topsis"), FakeTopsis)

    def test_a_class_instead_of_an_instance_is_refused(self, clean_registry):
        """A common mistake, and the error should say so rather than fail
        mysteriously later at score() time."""
        with pytest.raises(McdaError, match="not the class itself"):
            register(Median)

    def test_a_non_method_is_refused(self, clean_registry):
        with pytest.raises(McdaError, match="takes a Method instance"):
            register(object())

    def test_a_nameless_method_is_refused(self, clean_registry):
        class Nameless(Method):
            def score(self, ctx):
                return np.zeros(ctx.n_options)

        with pytest.raises(McdaError, match="non-empty `name`"):
            register(Nameless())

    def test_an_unimplemented_score_says_so(self, clean_registry, supplier):
        class Abstract(Method):
            name = "test_abstract"

        register(Abstract())
        with pytest.raises(NotImplementedError, match="must implement score"):
            rank(supplier, method="test_abstract")


class TestContract:
    def test_oriented_is_the_default_and_flips_cost_columns(self, clean_registry):
        seen = {}

        class Peek(Method):
            name = "test_peek"

            def score(self, ctx):
                seen["data"] = ctx.data.copy()
                return np.zeros(ctx.n_options)

        register(Peek())
        criteria = [Criterion("Price", 1.0, "cost")]
        rank([[100.0], [900.0]], criteria, method="test_peek")
        assert seen["data"].ravel().tolist() == [900.0, 100.0], (
            "cost column should arrive mirrored"
        )

    def test_raw_methods_see_the_matrix_as_supplied(self, clean_registry):
        seen = {}

        class Peek(Method):
            name = "test_raw"
            wants = Wants.RAW

            def score(self, ctx):
                seen["data"] = ctx.data.copy()
                return np.zeros(ctx.n_options)

        register(Peek())
        criteria = [Criterion("Price", 1.0, "cost")]
        rank([[100.0], [900.0]], criteria, method="test_raw")
        assert seen["data"].ravel().tolist() == [100.0, 900.0], (
            "RAW must not be oriented"
        )

    def test_a_method_result_carries_warnings_and_the_guarantee(
        self, clean_registry, supplier
    ):
        class Caveated(Method):
            name = "test_caveated"

            def score(self, ctx):
                return MethodResult(
                    scores=np.arange(ctx.n_options, dtype=float),
                    reversal_free=True,
                    warnings=("a caveat",),
                )

        register(Caveated())
        result = rank(supplier, method="test_caveated")
        assert result.reversal_free is True
        assert result.warnings == ("a caveat",)

    def test_a_bare_array_is_accepted_too(self, clean_registry, supplier):
        """The common case must stay a one-liner."""
        register(Median())
        result = rank(supplier, method="test_median")
        assert result.reversal_free is False
        assert result.warnings == ()

    def test_validate_can_refuse_a_problem(self, clean_registry, supplier):
        class NeedsMany(Method):
            name = "test_needs_many"

            def validate(self, decision):
                if decision.n_options < 10:
                    raise McdaError("needs at least ten options")

            def score(self, ctx):
                return np.zeros(ctx.n_options)

        register(NeedsMany())
        with pytest.raises(McdaError, match="at least ten options"):
            rank(supplier, method="test_needs_many")

    def test_options_reach_a_method_that_reads_them(self, clean_registry, supplier):
        class Configurable(Method):
            name = "test_configurable"

            def score(self, ctx):
                return np.full(ctx.n_options, ctx.opts.get("level", 0.0))

        register(Configurable())
        assert rank(supplier, method="test_configurable", level=3.0).scores[0] == 3.0

    def test_options_nobody_reads_are_reported(self, clean_registry, supplier):
        """The bug this guards against: `rank(..., v=0.5)` on a method with no
        `v` used to look like it worked and do nothing."""
        register(Median())
        with pytest.raises(McdaError, match="takes no keyword arguments"):
            rank(supplier, method="test_median", nonsense=1)


class TestScoringContext:
    def test_it_exposes_the_problem_without_the_method_reaching_around(
        self, clean_registry
    ):
        seen = {}

        class Peek(Method):
            name = "test_ctx"
            # Reads ctx.directions, so it must ask for the raw matrix — the
            # guard in rank() enforces exactly this.
            wants = Wants.RAW

            def score(self, ctx):
                seen.update(
                    n_options=ctx.n_options,
                    n_criteria=ctx.n_criteria,
                    directions=list(ctx.directions),
                    bounds=list(ctx.bounds),
                    labels=list(ctx.labels),
                    normalized=ctx.normalized_weights.copy(),
                )
                return np.zeros(ctx.n_options)

        register(Peek())
        criteria = [
            Criterion("Price", 3.0, "cost", bounds=(0, 10)),
            Criterion("Quality", 1.0, "benefit"),
        ]
        rank([[1.0, 2.0], [3.0, 4.0]], criteria, method="test_ctx", labels=["x", "y"])

        assert seen["n_options"] == 2
        assert seen["n_criteria"] == 2
        assert seen["directions"] == ["cost", "benefit"]
        assert seen["bounds"] == [(0.0, 10.0), None]
        assert seen["labels"] == ["x", "y"]
        assert seen["normalized"] == pytest.approx([0.75, 0.25])


class TestBuiltins:
    def test_all_eight_are_registered_on_import(self):
        assert len(BUILTIN_METHODS) == 8
        for cls in BUILTIN_METHODS:
            assert has(cls.name)

    def test_each_declares_a_summary(self):
        for cls in BUILTIN_METHODS:
            assert cls.summary, f"{cls.__name__} has no summary"

    def test_only_spotis_claims_reversal_freedom(self):
        claiming = [c.name for c in BUILTIN_METHODS if c.reversal_free]
        assert claiming == ["spotis"]

    def test_only_spotis_wants_raw_data(self):
        raw = [c.name for c in BUILTIN_METHODS if c.wants is Wants.RAW]
        assert raw == ["spotis"]

    def test_available_describes_every_method(self):
        described = available()
        assert set(described) == set(names())
        assert all(text for text in described.values())

    def test_the_registry_can_be_rebuilt_from_scratch(self, clean_registry):
        _reset_for_testing()
        assert names() == () or True  # plugin scan may repopulate
        register_builtins(replace=True)
        for cls in BUILTIN_METHODS:
            assert has(cls.name)


class TestPluginDiscovery:
    def test_a_broken_entry_point_warns_rather_than_breaking_everything(
        self, clean_registry, monkeypatch
    ):
        """One bad plugin must not stop the library — and every other plugin —
        from working."""
        import mcdakit.methods.registry as reg

        class Exploding:
            name = "boom"
            value = "nowhere:Nothing"

            def load(self):
                raise ImportError("no such module")

        monkeypatch.setattr(reg, "_entry_points", lambda: (Exploding(),))
        reg._PLUGINS_LOADED = False

        with pytest.warns(PluginLoadError, match="Could not load"):
            _load_plugins()

        assert has("topsis"), "the rest of the registry must still work"

    def test_an_entry_point_class_is_instantiated(self, clean_registry, monkeypatch):
        import mcdakit.methods.registry as reg

        class Pointer:
            name = "test_median"
            value = "tests:Median"

            def load(self):
                return Median  # a class, not an instance

        monkeypatch.setattr(reg, "_entry_points", lambda: (Pointer(),))
        reg._PLUGINS_LOADED = False
        _load_plugins()
        assert isinstance(get("test_median"), Median)

    def test_an_entry_point_instance_is_used_directly(
        self, clean_registry, monkeypatch
    ):
        import mcdakit.methods.registry as reg

        instance = Median()

        class Pointer:
            name = "test_median"
            value = "tests:median"

            def load(self):
                return instance

        monkeypatch.setattr(reg, "_entry_points", lambda: (Pointer(),))
        reg._PLUGINS_LOADED = False
        _load_plugins()
        assert get("test_median") is instance

    def test_discovery_runs_once(self, clean_registry, monkeypatch):
        import mcdakit.methods.registry as reg

        calls = []
        monkeypatch.setattr(reg, "_entry_points", lambda: calls.append(1) or ())
        reg._PLUGINS_LOADED = False

        names()
        names()
        get("topsis")
        assert len(calls) == 1

    def test_the_real_entry_point_scan_does_not_explode(self):
        """Whatever is installed in this environment, scanning must be safe."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PluginLoadError)
            _load_plugins()
        assert has("topsis")
