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
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
import cpmpy

# Import and run gallery preparation script
from prepare_galleries import prepare_galleries


# -- Project information -----------------------------------------------------

project = 'CPMpy'
copyright = '2026, Tias Guns'
author = 'Tias Guns'

# The full version, including alpha/beta/rc tags
release = '0.9.24'

# variables to be accessed from html
html_context = {
    'release': release,
    'webpage':  f'https://{project}.readthedocs.io/',
    # Pages nested under docs.rst, used by _templates/navbar-nav.html to
    # highlight the "Docs" top-nav tab as active (api/* is matched by prefix
    # instead, since there are too many pages to list here).
    'docs_pages': {
        'docs', 'modeling', 'summary', 'upgrading_to_v1', 'how_to_debug',
        'multiple_solutions', 'unsat_core_extraction', 'developers',
        'adding_solver', 'testing',
    },
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
    'myst_parser',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'sphinx_gallery.gen_gallery',
    # "nbsphinx",
    # "myst_nb"
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
napoleon_use_param = True
napoleon_use_rtype = True


todo_include_todos = True

# Strip shell prompts and Python REPL prompts from copied code snippets.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

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
# html_theme = 'sphinx_book_theme'
html_theme = "pydata_sphinx_theme"
html_favicon = "../logo/CPMpy_Icon_Blue.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    "custom.css"
]

html_js_files = [
    'custom.js',
]

html_theme_options = {
    "logo": {
        "image_light": "../logo/CPMpy_Icon_Blue.svg",
        "image_dark": "../logo/CPMpy_Icon_Blue.svg",
        "text": "CPMpy",
    },
    "github_url": "https://github.com/CPMpy/cpmpy",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/cpmpy/",
            "icon": "fa-brands fa-python",
        },
    ],
    # Top navbar shows "Docs" and "Examples" (index.rst's own toctree has
    # just those 2 entries) - add more top-level sections (e.g. "Playground")
    # by adding another entry to that toctree.
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "collapse_navigation": False,
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
}

# Prepare galleries before sphinx-gallery processes them
gallery_dirs = prepare_galleries()

import plotly.io as pio
pio.renderers.default = 'sphinx_gallery_png'

sphinx_gallery_conf = {
    'examples_dirs': [
        '_temp_galleries/basic',
        '_temp_galleries/csplib',
        #'_temp_galleries/tutorial_ijcai22',
    ],
    'gallery_dirs': [
        'auto_examples/basic',
        'auto_examples/csplib',
        #'auto_examples/tutorial_ijcai22',
    ],
    'filename_pattern': r'\.py$',  # Only process .py files with sphinx-gallery
    'ignore_pattern': r'__init__\.py',
    'download_all_examples': False,
    # 'show_download_links': False,
    'show_memory': False,
    'remove_config_comments': True,
    'expected_failing_examples': [],
    'plot_gallery': 'True',
    'abort_on_example_error': False,
    'example_extensions': {'.py', '.ipynb'},
    'image_scrapers': ('matplotlib', 'plotly.io._sg_scraper.plotly_sg_scraper'),
    # 'reset_modules': ('matplotlib', 'seaborn', 'plotly'),
    'capture_repr': ('_repr_html_', '__repr__', '__str__'),
}

html_title = "CPMpy documentation"

# man_pages = [
#     (master_doc, 'pysat', u'PySAT Documentation',
#      [author], 1)
# ]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'cpmpy'