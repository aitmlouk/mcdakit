"""The benchmark behind the README, pinned as a test.

The reversal table is this package's central claim. Running the benchmark only
by hand would let a regression sit unnoticed until someone happened to look.
"""

import warnings

import pytest
from benchmarks.reversal import random_problem, reverses, run

from mcdakit import METHODS, BoundsWarning


class TestBenchmark:
    def test_spotis_never_reverses(self):
        """The headline claim. Anything but zero is a bug, not a bad seed."""
        rates = run(trials=200, n_options=5, n_criteria=4, seed=20260826)
        assert rates["spotis"] == 0.0

    @pytest.mark.parametrize("seed", [1, 42, 99999])
    def test_spotis_never_reverses_on_any_seed(self, seed):
        rates = run(trials=60, n_options=5, n_criteria=4, seed=seed)
        assert rates["spotis"] == 0.0

    @pytest.mark.parametrize("shape", [(4, 3), (7, 5), (10, 2)])
    def test_spotis_never_reverses_at_any_size(self, shape):
        options, criteria = shape
        rates = run(trials=40, n_options=options, n_criteria=criteria, seed=7)
        assert rates["spotis"] == 0.0

    def test_the_other_methods_do_reverse(self):
        """If nothing reversed, the benchmark would prove nothing about
        SPOTIS — it would just mean the experiment was too easy."""
        rates = run(trials=200, n_options=5, n_criteria=4, seed=20260826)
        reversing = [m for m, rate in rates.items() if rate > 0.10]
        assert len(reversing) >= 4, rates

    def test_the_documented_ordering_holds(self):
        """The README says range-based methods suffer most. Pin the shape of
        the finding, not the exact percentages, which move with the seed."""
        rates = run(trials=200, n_options=5, n_criteria=4, seed=20260826)
        assert rates["vikor"] > rates["topsis"] > rates["saw"]
        assert rates["weighted_scoring"] > rates["saw"]

    def test_the_benchmark_uses_real_bounds(self, rng):
        """SPOTIS measured through its fallback would be measuring the wrong
        thing, so the generator must give every criterion real bounds."""
        decision = random_problem(rng, 5, 4)
        assert all(c.bounds is not None for c in decision.criteria)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BoundsWarning)
            assert reverses(decision, "spotis") is False

    @pytest.mark.parametrize("method", METHODS)
    def test_reverses_returns_a_verdict_for_every_method(self, rng, method):
        decision = random_problem(rng, 5, 4)
        assert isinstance(reverses(decision, method), bool)
