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

## Optional dependencies

The test suite (`tests/`) and the `SpeedComparision` tutorial notebook compare GRaTeR-JAX against the original `numpy` GRaTeR implementation in [VIP](https://github.com/vortex-exoplanet/VIP). `vip_hci` is not a core dependency; install the `dev` extra to get it together with `pytest`:

```bash
pip install "grater-jax[dev]"
```

The other extras are `docs` (Sphinx toolchain for building this documentation) and `webbpsf` (JWST PSF generation for the experimental Winnie-based forward model).

## Compatibility

GRaTeR-JAX is compatible with Python versions `>=3.10` and `<3.14`, and with GPU hardware supported by JAX. See the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html) for supported backends and platforms.

A full list of package dependencies is available in `pyproject.toml`.
