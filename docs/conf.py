# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'GRaTeR-JAX'
copyright = '2025, GRaTeR-JAX authors'
author = 'GRaTeR-JAX authors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',              # Markdown support
    'sphinx.ext.autodoc',       # Pull docstrings from code
    'sphinx.ext.autosummary',   # Generate API summary pages
    'sphinx.ext.napoleon',      # Google/NumPy style docstrings
    'numpydoc',                 # NumPy-style doc enhancements
    'nbsphinx',                 # Render Jupyter notebooks
    'sphinx_rtd_theme'          # Read the Docs theme
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Autodoc defaults — ensures methods are fully shown
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autosummary_imported_members = True

# If heavy imports fail on RTD, mock them
autodoc_mock_imports = [
    "jax", "jaxlib", "jaxopt",
    "astropy", "photutils",
    "stpsf", "webbpsf_ext", "pysiaf", "poppy",
    "h5py", "xarray",
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Newer Sphinx uses root_doc; default is "index" but set explicitly
root_doc = "index"
