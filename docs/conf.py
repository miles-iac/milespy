# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../"))

import milespy  # noqa: E402


# -- Project information -----------------------------------------------------

project = "MILESpy"
copyright = "2024, Isaac Alonso Asensio"
author = "Isaac Alonso Asensio"

# The full version, including alpha/beta/rc tags
release = milespy.__version__
version = milespy.__version__


# -- General configuration ---------------------------------------------------

autodoc_mock_imports = ["rcfile"]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "numpydoc",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["biblio.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "plain"

autosummary_generate = True
numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "specutils": ("https://specutils.readthedocs.io/en/stable/", None),
}

# Only executes the notebooks if we have availble the tutorial dataset
if Path("tutorials/stars.txt").exists() and os.environ.get("CI") != "true":
    nbsphinx_execute = "always"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_logo = "miles_header.jpg"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]
