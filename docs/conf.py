"""Sphinx configuration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "mcdakit"
author = "Addi Ait-Mlouk"
copyright = "2026, Addi Ait-Mlouk"

from mcdakit import __version__  # noqa: E402

version = release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = f"mcdakit {version}"
html_static_path = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Render "Attributes" sections as prose rather than as object descriptions;
# autodoc already emits an entry per attribute, and both would appear.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

# The extension classes are documented once, in the API reference. Sphinx
# otherwise reports every attribute twice: once for the class and once for the
# module it lives in.
suppress_warnings = ["ref.python", "autosectionlabel"]

nitpicky = False
