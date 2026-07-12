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
html_static_path = ["_static"]
html_favicon = "_static/solvax.svg"

# DOI targets are stable bibliography identifiers, but several publisher
# endpoints reject automated HEAD/GET requests with HTTP 403. Keep the links in
# rendered citations while excluding the resolver from linkcheck.
linkcheck_ignore = [r"https://doi.org/.*"]

intersphinx_mapping = {
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
}

autodoc_typehints = "description"
napoleon_google_docstring = True
# NamedTuple fields are already documented via napoleon Attributes sections;
# ivar rendering avoids duplicate autodoc entries for the same objects.
napoleon_use_ivar = True
