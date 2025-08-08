# GRaTeR-JAX

**GRaTeR-JAX** is a machine learning JAX-based implementation of the **Generalized Radial Transporter (GRaTeR)** framework, designed for modeling scattered light disks in protoplanetary systems. This repository provides tools for forward modeling, optimization, and parameter estimation of scattered light disk images using JAX's accelerated computations.

<img src="https://github.com/user-attachments/assets/c10f45e8-5449-4891-b6a7-33954cf6d954" width="300">

## Features

- **JAX-Based Optimization**: Leverages JAX for fast, GPU/TPU-accelerated disk modeling.
- **Scattered Light Disk Modeling**: Implements physical models of exoplanetary debris disks.
- **Differentiable Framework**: Enables gradient-based optimization and probabilistic inference.
- **Integration with Webbpsf**: Supports PSF convolution for telescope observations.

## Installation

To install GRaTeR-JAX and its dependencies, run:

```sh
git clone https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX.git
cd GRaTeR-JAX
conda env create -f environment.yml
```

Make sure you have JAX installed with the correct backend for your hardware:

```sh
pip install --upgrade "jax[cpu]"  # or "jax[cuda]" for GPU
```

## Usage

### Refer to tutorials/DiskTutorial.ipynb for building basic disk models and fitting them to images.
### Refer to tutorials/FitsTutorial.ipynb for analyzing fits files, fitting disks to them, and running MCMC tests.

Information about the disk and misc parameters can be found in objective_functions.py. Information about the
scattering phase function and point spread function parameters can be found in SLD_utils.py. Documentation is 
under construction at [grater-jax.readthedocs.io](https://grater-jax.readthedocs.io/en/latest/).

Check out [GRaTeR Image Generator](https://scattered-light-disks.vercel.app) to visualize how each of the parameters affect the disk model!

## Repository Structure

```
GRaTeR-JAX/
│── disk_model/            # Code for disk modeling
│── optimization/          # Tools for statistical optimization and analysis
|── tutorials/             # Tutorial Jupyter notebooks
│── webbpsf-data           # PSF data for various instruments
│── PSFs/                  # PSF data for the disk model
│── environment.yml        # Dependencies
│── requirements.txt       # Pip dependencies
│── README.md              # This document
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

Developed by the **UCSB Exoplanet Polarimetry Lab**. This work is inspired by previous implementations of GRaTeR and advances in JAX-based differentiable modeling. Additional thanks to Kellen Lawson for developing the Winnie package that this framework uses to model JWST PSFs.

---
