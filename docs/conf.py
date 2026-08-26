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
    "sphinx_copybutton",
    "sphinx_design",
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"

# Syntax highlighting, chosen per mode rather than left to one default that
# has to work on both. Furo swaps between these with the theme toggle.
pygments_style = "friendly"
pygments_dark_style = "github-dark"

html_theme_options = {
    # A single accent, used for links, the active nav item and admonition
    # rules. Teal reads clearly on both grounds without vibrating against the
    # dark background the way a saturated blue does.
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0f766e",
        "color-api-name": "#9a3412",
        "color-api-pre-name": "#9a3412",
        "font-stack": (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, Roboto, "
            "'Helvetica Neue', Arial, sans-serif"
        ),
        "font-stack--monospace": (
            "'SF Mono', SFMono-Regular, ui-monospace, 'JetBrains Mono', "
            "Menlo, Consolas, monospace"
        ),
    },
    "dark_css_variables": {
        # Lifted and desaturated: the light-mode teal is too dark to read
        # against a near-black background.
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#5eead4",
        "color-api-name": "#fdba74",
        "color-api-pre-name": "#fdba74",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/aitmlouk/mcdakit",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/aitmlouk/mcdakit",
            "class": "",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16" height="1em" '
                'width="1em"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-"
                ".48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 "
                "1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-"
                "1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-"
                ".36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
                ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 "
                "1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 "
                "3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15"
                '.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z">'
                "</path></svg>"
            ),
        },
    ],
}

# Copy buttons on code blocks. The prompt filter means copying a shell example
# does not also copy the `$`, and copying a doctest skips the `>>>`.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

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
