"""Sphinx configuration for the solvax documentation."""

import solvax

project = "solvax"
copyright = "2026, UW Plasma group"
author = "UW Plasma group"
release = solvax.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["references.bib"]
myst_enable_extensions = ["dollarmath", "amsmath"]

html_theme = "furo"
html_title = "solvax"

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = True
# NamedTuple fields are already documented via napoleon Attributes sections;
# ivar rendering avoids duplicate autodoc entries for the same objects.
napoleon_use_ivar = True
