# Installation

## Installing via PyPI

We recommend using a Conda environment (or another virtual environment) when installing GRaTeR-JAX to ensure that dependencies are properly isolated and compatible.

For a standard installation, create a new environment with a supported Python version and run:

```bash
pip install grater-jax
```

Make sure JAX is installed with the appropriate backend for your hardware:

```bash
pip install --upgrade "jax[cpu]"   # or "jax[cuda]" for GPU support
```

## Compatibility

GRaTeR-JAX is compatible with Python versions `>=3.9` and `<3.13`, and with GPU hardware supported by JAX. See the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html) for supported backends and platforms.

A full list of package dependencies is available in `pyproject.toml`.
