# -- Path setup --------------------------------------------------------------
import os
import sys

# Add the project root (one level up from docs/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -- Project information -----------------------------------------------------
project = 'GRaTeR-JAX'
author = 'GRaTeR-JAX authors'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',              # Markdown support
    'sphinx.ext.autodoc',       # Pull docstrings from code
    'sphinx.ext.autosummary',   # Generate API summary pages
    'sphinx.ext.napoleon',      # Google/NumPy style/Numpy style docstrings
    'nbsphinx',                 # Render Jupyter notebooks
    'sphinx_rtd_theme',         # Read the Docs theme
]
autosummary_generate = True

# MyST config (handy but minimal)
myst_enable_extensions = [
    "colon_fence",              # ::: fences
    "deflist",                  # definition lists
]

# Let autosummary create API stubs
autosummary_generate = True
autosummary_imported_members = True
autosummary_generate_overwrite = True

templates_path = ['_templates']
# Don’t pick up checkpoint notebooks
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
# Keep type hints out of signatures to avoid odd wrappers causing issues
autodoc_typehints = "none"

# If heavy imports fail on RTD, mock them
autodoc_mock_imports = [
    "astropy", "photutils",
    "stpsf", "webbpsf_ext", "pysiaf", "poppy",
    "h5py", "xarray",
    "numpyro", "cupy",
    "corner", "arviz", "emcee",
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
# Avoid warning if you don't have docs/_static
html_static_path = []  # change to ['_static'] if you add that folder

# Root doc for Sphinx >= 5
root_doc = "index"

# -- nbsphinx options --------------------------------------------------------
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'  # Don’t run notebooks on RTD

source_suffix = {
    '.rst': 'restructuredtext',
    '.md':  'myst',
}
exclude_patterns += ['**.ipynb_checkpoints']