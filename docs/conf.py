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
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import cpmpy


# -- Project information -----------------------------------------------------

project = 'CPMpy'
copyright = '2024, Tias Guns'
author = 'Tias Guns'

# The full version, including alpha/beta/rc tags
release = '0.9.19'

# variables to be accessed from html
html_context = {
    'release': release,
    'webpage':  f'https://{project}.readthedocs.io/'
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'm2r2',
    'sphinx_rtd_theme',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver'
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

source_suffix =  ['.rst', '.md']
# source_suffix =  '.rst'

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# Autodoc settings
autodoc_default_flags = ['members', 'special-members']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# man_pages = [
#     (master_doc, 'pysat', u'PySAT Documentation',
#      [author], 1)
# ]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'cpmpy'