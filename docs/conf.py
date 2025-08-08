# Configuration file for the Sphinx documentation builder.

project = "GRaTeR-JAX"
author = "Mihir Kondapalli"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"