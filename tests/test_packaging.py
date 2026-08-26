"""Packaging invariants.

Metadata drifts silently: a version bumped in one file and not another, a
method added to the registry but not the README table. Each of these is
invisible until someone reads the wrong number and believes it.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import mcdakit

ROOT = Path(__file__).resolve().parents[1]


class TestVersion:
    def test_the_installed_metadata_matches_the_package(self):
        """pyproject reads the version from __init__, so these cannot
        disagree unless the build backend config breaks."""
        from importlib.metadata import version

        assert version("mcdakit") == mcdakit.__version__

    def test_citation_declares_the_same_version(self):
        """CITATION.cff cannot read the package, so it is the one copy that
        can drift. Researchers cite what it says."""
        text = (ROOT / "CITATION.cff").read_text()
        declared = re.search(r"^version:\s*(\S+)", text, re.M)
        assert declared, "CITATION.cff has no version"
        assert declared.group(1) == mcdakit.__version__

    def test_it_looks_like_a_version(self):
        assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-.].+)?", mcdakit.__version__)


class TestDocumentedMethods:
    def test_every_registered_method_is_in_the_readme_table(self):
        """A method nobody can find is a method nobody uses."""
        readme = (ROOT / "README.md").read_text()
        for name in mcdakit.METHODS:
            assert f"`{name}`" in readme, (
                f"{name} is registered but absent from the README"
            )

    def test_the_readme_claims_no_methods_that_do_not_exist(self):
        """The reverse drift: a method removed but still advertised."""
        readme = (ROOT / "README.md").read_text()
        table = re.findall(r"^\| `(\w+)` \|", readme, re.M)
        assert table, "the method table has changed shape"
        for name in table:
            assert name in mcdakit.METHODS, (
                f"the README documents {name!r}, which is not registered"
            )

    def test_every_method_declares_a_citation(self):
        """This library is read by people who know the papers."""
        from mcdakit.methods import BUILTIN_METHODS

        uncited = [
            cls.name
            for cls in BUILTIN_METHODS
            if not cls.citation
            and cls.name not in {"simple_scoring", "weighted_scoring"}
        ]
        assert not uncited, f"no citation for: {uncited}"


class TestNoOdoo:
    def test_the_source_is_free_of_odoo(self):
        """Rule 1 of the original brief: if this only works inside Odoo, it
        failed. The brief itself is exempt — it is a historical document that
        explains the port."""
        for path in (ROOT / "src").rglob("*.py"):
            assert "odoo" not in path.read_text().lower(), path


class TestEntryPointGroup:
    def test_the_advertised_group_matches_what_the_docs_tell_plugins(self):
        """A typo here means every third-party plugin silently fails to
        load, with nothing to indicate why."""
        from mcdakit.methods.registry import ENTRY_POINT_GROUP

        assert ENTRY_POINT_GROUP == "mcdakit.methods"
        for doc in ("docs/extending.md", "README.md"):
            assert ENTRY_POINT_GROUP in (ROOT / doc).read_text(), doc


class TestConformanceCli:
    @pytest.mark.parametrize(
        "method",
        [
            "SimpleScoring",
            "WeightedScoring",
            "Saw",
            "Topsis",
            "Vikor",
            "Electre",
            "Promethee",
            "Spotis",
        ],
    )
    def test_every_builtin_passes_from_the_command_line(self, method):
        """CI runs exactly this. Holding contributors to a check the library
        itself fails would be indefensible, so it is tested here too."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mcdakit.testing",
                f"mcdakit.methods:{method}",
                "--no-strict",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 0, result.stdout + result.stderr


class TestDocumentation:
    def test_every_public_name_appears_in_the_api_reference(self):
        """An exported name absent from the docs is one nobody can discover.

        Matched against docs/api.md rather than a built site, so the check
        needs no Sphinx run; names may be written bare or qualified.
        """
        api = (ROOT / "docs" / "api.md").read_text()
        missing = [
            name
            for name in mcdakit.__all__
            if not name.startswith("__")
            and name not in api
            and f"mcdakit.{name}" not in api
        ]
        assert not missing, f"undocumented in docs/api.md: {missing}"

    def test_every_public_callable_has_a_docstring(self):
        import inspect

        undocumented = [
            name
            for name in mcdakit.__all__
            if (
                inspect.isfunction(obj := getattr(mcdakit, name, None))
                or inspect.isclass(obj)
            )
            and not (obj.__doc__ or "").strip()
        ]
        assert not undocumented

    def test_the_guides_are_wired_into_the_toctree(self):
        """A page absent from index.md builds but is unreachable."""
        index = (ROOT / "docs" / "index.md").read_text()
        for page in ("quickstart", "stability", "extending", "api"):
            assert page in index, f"{page} is not in the toctree"
