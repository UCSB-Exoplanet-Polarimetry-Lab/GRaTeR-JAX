# -- Project info -----------------------------------------------------
project = "GRaTeR-JAX"
author = "GRaTeR-JAX authors"
release = "0.1.0"

# -- General config ---------------------------------------------------
extensions = [
    "myst_parser",              # allow Markdown
    "sphinx.ext.autodoc",       # pull docstrings
    "sphinx.ext.autosummary",   # generate API stubs
    "sphinx.ext.napoleon",      # Google/NumPy style docstrings
]

autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []

# If imports are heavy or require GPUs / external data, mock them here
autodoc_mock_imports = [
    "jax", "jaxlib", "jaxopt",
    "astropy", "photutils",
    "stpsf", "webbpsf_ext", "pysiaf", "poppy",
    "h5py", "xarray",
]

# HTML theme
html_theme = "sphinx_rtd_theme"  # or "alabaster"
html_static_path = ["_static"]

# Newer Sphinx uses root_doc; default is "index" but set explicitly
root_doc = "index"

# Make sure your package can be imported by autodoc
import os, sys
# If your code is in the repo root as packages (e.g., disk_model/, optimization/):
sys.path.insert(0, os.path.abspath(".."))
# If you use a src/ layout, use this instead:
# sys.path.insert(0, os.path.abspath("../src"))
