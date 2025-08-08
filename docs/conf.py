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
    # 'numpydoc',               # Disabled: conflicts with wrapped/mocked JAX objects on Py3.13
    'nbsphinx',                 # Render Jupyter notebooks
    'sphinx_rtd_theme'          # Read the Docs theme
]

autosummary_generate = True
autosummary_imported_members = True

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

# If heavy imports fail on RTD, mock them
autodoc_mock_imports = [
    "jax", "jaxlib", "jaxopt",
    "astropy", "photutils",
    "stpsf", "webbpsf_ext", "pysiaf", "poppy",
    "h5py", "xarray",
    "numpyro",  # needed because optimization.* may import it
]

# Napoleon tweaks (let it fully replace numpydoc functionality)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
# Set to [] if you don't have a _static directory to avoid warnings
html_static_path = []  # or create docs/_static and use ['_static']

# Newer Sphinx uses root_doc; default is "index" but set explicitly
root_doc = "index"
