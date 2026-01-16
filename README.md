# GRaTeR-JAX

[![Run Tests](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX/actions/workflows/tests.yml/badge.svg)](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX/actions/workflows/tests.yml)

**GRaTeR-JAX** is a machine learning JAX-based implementation of the **Generalized Radial Transporter (GRaTeR)** framework [(Augereau+ 1999)](https://arxiv.org/abs/astro-ph/9906429), designed for modeling scattered-light images of debris disks. This repository provides tools for forward modeling, optimization, and parameter estimation of scattered-light disk images using JAX's accelerated computations.

<img src="https://github.com/user-attachments/assets/c10f45e8-5449-4891-b6a7-33954cf6d954" width="300">

## Features

- **JAX-Based Optimization**: Leverages JAX for fast, GPU/TPU-accelerated disk modeling.
- **Scattered Light Disk Modeling**: Implements physical models of exoplanetary debris disks.
- **Differentiable Framework**: Enables gradient-based optimization and probabilistic inference.
- **Integration with Webbpsf**: Supports JWST PSF convolutions for forward modeling.

## Installation

To install GRaTeR-JAX and its dependencies, create a new Conda environment with Python and run:

```sh
pip install grater-jax
```

Make sure you have JAX installed with the correct backend for your hardware:

```sh
pip install --upgrade "jax[cpu]"  # or "jax[cuda]" for GPU
```

Highly recommended to install this on a fresh environment, just to be safe.

## Usage

Refer to the documentation at [grater-jax.readthedocs.io](https://grater-jax.readthedocs.io/en/latest/).

Check out [GRaTeR Image Generator](https://scattered-light-disks.vercel.app) to visualize how each of the parameters affect the disk model!

## Repository Structure

```
GRaTeR-JAX/
│── grater-jax/       # Package root for grater-jax
   │── disk_model/    # Code for disk modeling
   │── optimization/  # Tools for statistical optimization and analysis
|── docs/             # Documenation and tutorial notebooks
│── pyproject.toml    # Installation file
│── README.md
```

## Contributing

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```sh
   git checkout -b feature-branch
   ```
3. Commit your changes and push to your fork.
4. Open a pull request.

## Acknowledgments

Developed by the **UCSB Exoplanet Polarimetry Lab**. This work is inspired by previous implementations of GRaTeR and advances in JAX-based differentiable modeling.

---
