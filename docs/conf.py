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
copyright = '2025, Tias Guns'
author = 'Tias Guns'

# The full version, including alpha/beta/rc tags
release = '0.9.24'

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
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    "sphinx_immaterial",
    # "sphinx_immaterial.theme_result",
    # "sphinx_immaterial.kbd_keys",
    # "sphinx_immaterial.apidoc.format_signatures",
    # "sphinx_immaterial.apidoc.json.domain",
    # "sphinx_immaterial.apidoc.python.apigen",
    # "sphinx_immaterial.graphviz",
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
html_theme = "sphinx_immaterial"

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

templates_path = ["_templates"]

html_theme_options = {
    "repository_url": "https://github.com/CPMpy/cpmpy",
    "use_repository_button": True,
    "site_url": "https://cpmpy.readthedocs.io/",
    "repo_url": "https://github.com/CPMpy/cpmpy",
    # "edit_uri": "blob/main/docs",
    "features": [
        "navigation.expand",
        # "navigation.tabs",
        # "navigation.tabs.sticky",
        # "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        "navigation.top",
        "navigation.footer",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        "content.tabs.link",
        "content.code.copy",
        "content.action.edit",
        "content.action.view",
        "content.tooltips",
        "announce.dismiss",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            # "toggle": {
            #     "icon": "material/brightness-auto",
            #     "name": "Switch to light mode",
            # },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "white",
            "accent": "light-blue",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "deep-orange",
            "accent": "lime",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
    "toc_title_is_page_title": False,

    "social": [
        {
            "icon": "fontawesome/brands/github",
            "link": "https://github.com/CPMpy/cpmpy",
            "name": "Source on github.com",
        },
        {
            "icon": "fontawesome/brands/python",
            "link": "https://pypi.org/project/cpmpy/",
        },
    ]
    
}

html_title = "CPMpy documentation"

# man_pages = [
#     (master_doc, 'pysat', u'PySAT Documentation',
#      [author], 1)
# ]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'cpmpy'