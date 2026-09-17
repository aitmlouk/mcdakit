"""ARAS, COCOSO, CODAS, EDAS, MABAC and MARCOS.

Each is validated against reference values that two independent libraries
produce *and agree on*, which is what makes them usable as ground truth rather
than one library's opinion. Agreement is then checked across random problems,
since a single matching example can be a coincidence of the fixture.

Beyond agreement, each method's defining property is asserted directly. A
method can agree with another implementation on one problem and still be
wrong; it cannot rank a dominated alternative first and be right.
"""

import numpy as np
import pytest

from mcdakit import METHODS, Criterion, Decision, McdaError, rank
from mcdakit.methods.reference_point import (
    aras,
    cocoso,
    codas,
    edas,
    mabac,
    marcos,
)
from mcdakit.testing import conformance_report

MATRIX = np.array(
    [
        [2.75, 7.0, 14.0, 8.0],
        [2.90, 8.5, 16.0, 8.0],
        [3.40, 9.0, 11.0, 7.0],
        [2.20, 3.0, 32.0, 2.0],
    ]
)
WEIGHTS = np.array([0.40, 0.25, 0.20, 0.15])
DIRECTIONS = ["cost", "benefit", "cost", "benefit"]
LABELS = ["Option 1", "Option 2", "Option 3", "Option 4"]

#: Values pymcdm 1.4.0 and pyrepo-mcda 0.1.15 both produce, independently.
#: MABAC is pymcdm only, pyrepo-mcda exposing it differently.
REFERENCE = {
    "aras": [0.82099, 0.82585, 0.84476, 0.58147],
    "cocoso": [3.44681, 3.42054, 2.80552, 1.21187],
    "codas": [0.20733, 0.28018, 0.43874, -0.92625],
    "edas": [0.79192, 0.84404, 0.89386, 0.24928],
    "mabac": [0.16230, 0.15575, 0.03254, -0.14246],
    "marcos": [0.72144, 0.72624, 0.73767, 0.51771],
}

FUNCTIONS = {
    "aras": aras,
    "cocoso": cocoso,
    "codas": codas,
    "edas": edas,
    "mabac": mabac,
    "marcos": marcos,
}


class TestAgainstIndependentImplementations:
    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_it_reproduces_the_reference_values(self, name):
        scores = FUNCTIONS[name](MATRIX, WEIGHTS, DIRECTIONS)
        assert scores == pytest.approx(REFERENCE[name], abs=1e-5)

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_it_agrees_on_random_problems(self, name, seed):
        """Checked against pymcdm where it is installed. A single agreeing
        example proves little; agreement across shapes, weightings and
        direction mixtures is what makes the implementation credible."""
        pytest.importorskip("pymcdm", reason="dev-only cross-check")
        from pymcdm import methods as pm

        rng = np.random.default_rng(seed)
        n_options = int(rng.integers(3, 8))
        n_criteria = int(rng.integers(3, 6))
        matrix = rng.uniform(0.5, 10, size=(n_options, n_criteria))
        weights = rng.dirichlet(np.ones(n_criteria))
        types = np.array([1 if rng.random() > 0.4 else -1 for _ in range(n_criteria)])
        directions = ["cost" if t == -1 else "benefit" for t in types]

        ours = FUNCTIONS[name](matrix, weights, directions)
        theirs = np.asarray(
            getattr(pm, name.upper())()(matrix, weights, types), dtype=float
        )
        assert ours == pytest.approx(theirs, abs=1e-6)


class TestDefiningProperties:
    """What each method promises, asserted directly rather than inferred from
    agreement with another implementation."""

    def test_aras_scores_are_utility_ratios_at_most_one(self):
        """ARAS scores each alternative against a constructed optimum, so no
        real alternative can exceed it."""
        scores = aras(MATRIX, WEIGHTS, DIRECTIONS)
        assert np.all(scores > 0) and np.all(scores <= 1.0 + 1e-12)

    def test_mabac_sign_separates_above_from_below_the_border(self):
        """MABAC's zero point is interpretable: positive is above the
        geometric mean of the field, negative below."""
        scores = mabac(MATRIX, WEIGHTS, DIRECTIONS)
        assert np.any(scores > 0) and np.any(scores < 0)

    def test_edas_measures_distance_from_the_average(self):
        """An alternative equal to the column average on every criterion
        should sit mid-scale, neither rewarded nor penalised."""
        matrix = np.array([[10.0, 10.0], [2.0, 2.0], [6.0, 6.0]])
        scores = edas(matrix, np.array([0.5, 0.5]), ["benefit", "benefit"])
        assert scores[0] > scores[2] > scores[1]

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_a_dominant_alternative_ranks_first(self, name):
        """The one result no method may get wrong."""
        matrix = np.vstack([MATRIX, [2.10, 9.5, 10.0, 9.0]])
        scores = FUNCTIONS[name](matrix, WEIGHTS, DIRECTIONS)
        assert int(np.argmax(scores)) == len(matrix) - 1

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_cheaper_is_better_on_a_cost_criterion(self, name):
        matrix = np.array([[10.0, 5.0], [20.0, 5.0]])
        scores = FUNCTIONS[name](matrix, np.array([0.5, 0.5]), ["cost", "benefit"])
        assert scores[0] > scores[1], (
            f"{name} ranked the more expensive alternative higher"
        )


class TestDegenerateInput:
    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_all_zero_weights_return_zeros(self, name):
        scores = FUNCTIONS[name](MATRIX, np.zeros(4), DIRECTIONS)
        assert scores == pytest.approx(np.zeros(4))

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_a_flat_criterion_stays_finite(self, name):
        matrix = np.array([[5.0, 1.0], [5.0, 9.0], [5.0, 4.0]])
        scores = FUNCTIONS[name](matrix, np.array([0.5, 0.5]), ["benefit", "benefit"])
        assert np.all(np.isfinite(scores))

    def test_aras_refuses_a_non_positive_value(self):
        """Cost criteria are inverted before normalisation."""
        matrix = MATRIX.copy()
        matrix[0, 0] = 0.0
        with pytest.raises(McdaError, match="strictly positive"):
            aras(matrix, WEIGHTS, DIRECTIONS)

    @pytest.mark.parametrize("name", ["cocoso", "codas", "edas", "mabac", "marcos"])
    def test_a_zero_is_tolerated_where_the_method_allows_it(self, name):
        matrix = MATRIX.copy()
        matrix[0, 1] = 0.0
        assert np.all(np.isfinite(FUNCTIONS[name](matrix, WEIGHTS, DIRECTIONS)))


class TestParameters:
    def test_cocoso_lambda_changes_the_compromise(self):
        low = cocoso(MATRIX, WEIGHTS, DIRECTIONS, lambda_=0.1)
        high = cocoso(MATRIX, WEIGHTS, DIRECTIONS, lambda_=0.9)
        assert not np.allclose(low, high)

    def test_codas_tau_changes_the_tie_threshold(self):
        """Raising tau treats more pairs as tied on Euclidean distance, so
        the Taxicab term decides more of them."""
        tight = codas(MATRIX, WEIGHTS, DIRECTIONS, tau=0.001)
        loose = codas(MATRIX, WEIGHTS, DIRECTIONS, tau=10.0)
        assert not np.allclose(tight, loose)


class TestThroughTheApi:
    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_it_is_registered(self, name):
        assert name in METHODS

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_rank_reproduces_the_reference_scores(self, name):
        criteria = [
            Criterion(n, float(w), d)
            for n, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]
        result = rank(Decision(MATRIX, criteria, LABELS), method=name)
        assert result.scores == pytest.approx(REFERENCE[name], abs=1e-5)

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_it_passes_the_conformance_suite(self, name):
        from mcdakit.methods.registry import get

        failures = {k: v for k, v in conformance_report(get(name)).items() if v}
        assert not failures, failures

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_it_takes_the_matrix_as_measured(self, name):
        """All six resolve criterion direction themselves; orienting first
        would apply the correction twice."""
        from mcdakit.methods.base import Wants
        from mcdakit.methods.registry import get

        assert get(name).wants is Wants.RAW

    @pytest.mark.parametrize("name", sorted(FUNCTIONS))
    def test_sensitivity_works_on_it(self, name):
        from mcdakit import sensitivity

        criteria = [
            Criterion(n, float(w), d)
            for n, w, d in zip(
                ("Price", "Quality", "Lead time", "Support"), WEIGHTS, DIRECTIONS
            )
        ]
        report = sensitivity(rank(Decision(MATRIX, criteria, LABELS), method=name))
        assert report["level"] in {"fragile", "moderate", "robust", "immovable"}
