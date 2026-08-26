"""The public API: rank(), compare_methods(), agreement(), reversal_check()."""

import numpy as np
import pytest

import mcdakit
from mcdakit import (
    METHODS,
    Criterion,
    Decision,
    McdaError,
    agreement,
    compare_methods,
    rank,
    reversal_check,
)


class TestRankSignature:
    def test_it_accepts_a_decision(self, supplier):
        assert (
            rank(supplier).winner
            == rank(supplier.matrix, supplier.criteria, labels=supplier.labels).winner
        )

    def test_passing_both_a_decision_and_criteria_is_refused(self, supplier):
        """Silently ignoring one of them would hide a real mistake."""
        with pytest.raises(McdaError, match="not both"):
            rank(supplier, supplier.criteria)

    def test_criteria_are_required_for_a_bare_matrix(self):
        with pytest.raises(McdaError, match="criteria are required"):
            rank([[1, 2]])

    def test_method_specific_arguments_reach_the_method(self, supplier):
        """Regression: kwargs were dropped on the way in, so `v` was silently
        ignored and every VIKOR call used the default."""
        assert not np.allclose(
            rank(supplier, method="vikor", v=0.0).scores,
            rank(supplier, method="vikor", v=1.0).scores,
        )

    def test_an_argument_the_method_does_not_take_is_an_error(self, supplier):
        """`topsis` has no `v`. Accepting it silently is how the original
        kwargs bug hid: the call looked like it worked and did nothing."""
        with pytest.raises(McdaError, match="takes no keyword arguments"):
            rank(supplier, method="topsis", v=0.5)

    def test_an_argument_a_method_does_take_is_rejected_when_misspelled(self, supplier):
        """`vikor` reads options, so a wrong name reaches the function and
        surfaces as its TypeError — still an error, not a silent no-op."""
        with pytest.raises(TypeError, match="unexpected keyword"):
            rank(supplier, method="vikor", vee=0.5)


class TestCompareMethods:
    def test_it_runs_every_method_by_default(self, procurement):
        results = compare_methods(procurement)
        assert set(results) == set(METHODS)
        assert all(r.method == name for name, r in results.items())

    def test_a_subset_can_be_requested(self, procurement):
        results = compare_methods(procurement, methods=["topsis", "spotis"])
        assert set(results) == {"topsis", "spotis"}

    def test_an_unknown_method_is_refused_before_any_work(self, procurement):
        with pytest.raises(McdaError, match="Unknown method"):
            compare_methods(procurement, methods=["topsis", "nonsense"])


class TestAgreement:
    def test_it_counts_the_winners(self, procurement):
        summary = agreement(compare_methods(procurement))
        assert sum(summary["winners"].values()) == len(METHODS)
        assert summary["of"] == len(METHODS)
        assert summary["votes"] == summary["winners"][summary["consensus"]]

    def test_disagreement_is_surfaced_not_hidden(self, procurement):
        """When methods split, that split is the finding."""
        summary = agreement(compare_methods(procurement))
        assert summary["unanimous"] is False
        assert len(summary["winners"]) > 1

    def test_unanimity_is_reported_when_it_holds(self, supplier_criteria):
        matrix = [[9, 9, 9, 9], [5, 5, 5, 5], [1, 1, 1, 1]]
        results = compare_methods(
            matrix,
            supplier_criteria,
            labels=["Best", "Mid", "Worst"],
            methods=["weighted_scoring", "saw", "topsis"],
        )
        summary = agreement(results)
        assert summary["unanimous"] is True
        assert summary["consensus"] == "Best"


class TestReversalCheck:
    def test_it_never_removes_the_winner(self, procurement):
        """Removing the winner asks a different question — who wins now —
        rather than whether the losers' order was stable."""
        check = reversal_check(procurement, method="weighted_scoring")
        winner = rank(procurement, method="weighted_scoring").winner
        assert all(case["removed"] != winner for case in check["cases"])

    def test_a_case_records_both_orders(self, procurement):
        check = reversal_check(procurement, method="weighted_scoring")
        assert check["reversed"]
        case = check["cases"][0]
        assert case["before"] != case["after"]
        assert sorted(case["before"]) == sorted(case["after"])

    def test_spotis_reports_no_cases(self, procurement):
        assert reversal_check(procurement, method="spotis") == {
            "reversed": False,
            "cases": [],
        }


class TestPackage:
    def test_everything_exported_exists(self):
        for name in mcdakit.__all__:
            assert hasattr(mcdakit, name), name

    def test_it_has_a_version(self):
        assert mcdakit.__version__ == "0.1.0"

    def test_no_odoo_anywhere(self):
        """Rule 1 of the brief: if this only works inside Odoo, it failed."""
        import pathlib

        root = pathlib.Path(mcdakit.__file__).parent
        for path in root.rglob("*.py"):
            assert "odoo" not in path.read_text().lower(), path

    def test_numpy_is_the_only_third_party_import(self):
        """The one-dependency promise is a large part of the pitch."""
        import pathlib
        import re

        root = pathlib.Path(mcdakit.__file__).parent
        allowed = {"numpy", "mcdakit"}
        stdlib = {
            "__future__",
            "dataclasses",
            "typing",
            "warnings",
            "argparse",
            "sys",
            "abc",
            "math",
            "collections",
            "pathlib",
            "enum",
        }
        for path in root.rglob("*.py"):
            for line in path.read_text().splitlines():
                match = re.match(r"(?:from|import)\s+([\w.]+)", line)
                if not match:
                    continue
                top = match.group(1).split(".")[0]
                if top.startswith("_") or line.strip().startswith("from ."):
                    continue
                assert top in allowed | stdlib, f"{path}: {line.strip()}"


class TestDegenerate:
    def test_one_criterion_still_ranks(self):
        result = rank(
            [[1.0], [5.0], [3.0]],
            [Criterion("Only", 1.0, bounds=(0, 10))],
            method="spotis",
            labels=["a", "b", "c"],
        )
        assert result.winner == "b"

    @pytest.mark.parametrize("method", METHODS)
    def test_identical_options_tie_rather_than_ordering_arbitrarily(self, method):
        criteria = [
            Criterion("A", 1.0, bounds=(0, 10)),
            Criterion("B", 1.0, bounds=(0, 10)),
        ]
        result = rank(
            [[5.0, 5.0], [5.0, 5.0]], criteria, method=method, labels=["x", "y"]
        )
        assert result.scores[0] == pytest.approx(result.scores[1])
        assert list(result.ranks) == [1, 1]

    def test_a_criterion_with_zero_weight_is_ignored(self, supplier_criteria):
        """Muting one criterion is a legitimate what-if."""
        muted = [
            Criterion(c.name, 0.0 if c.name == "Support" else c.weight)
            for c in supplier_criteria
        ]
        matrix = [[9, 5, 7, 0], [6, 9, 8, 10], [7, 7, 6, 5]]
        without_support = Decision(
            [row[:3] for row in matrix], muted[:3], ["A", "B", "C"]
        )
        assert rank(matrix, muted, labels=["A", "B", "C"]).scores == pytest.approx(
            rank(without_support).scores
        )
