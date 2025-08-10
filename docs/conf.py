# -- Path setup --------------------------------------------------------------
import os
import sys

# Add the project root (one level up from docs/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    'nbsphinx',                 # Render Jupyter notebooks
    'sphinx_rtd_theme'          # Read the Docs theme
]

autosummary_generate = True
autosummary_imported_members = True
autosummary_generate_overwrite = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# If heavy imports fail on RTD, mock them
autodoc_mock_imports = [
    "jax", "jaxlib", "jaxopt",
    "astropy", "photutils",
    "stpsf", "webbpsf_ext", "pysiaf", "poppy",
    "h5py", "xarray",
    "numpyro",
]

# Napoleon tweaks
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = []  # or ['_static'] if you have static assets

# Root doc for Sphinx >= 5
root_doc = "index"

# -- nbsphinx options --------------------------------------------------------
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'  # Don’t run notebooks on RTD, just render them