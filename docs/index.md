# GRaTeR-JAX
**GPU-accelerated modeling of scattered-light disks**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c10f45e8-5449-4891-b6a7-33954cf6d954" width="220">
</p>

**GRaTeR-JAX** is a Python library for forward modeling of scattered-light images of circumstellar debris disks.  
It builds upon the *Generalized Radial Transporter (GRaTeR)* framework [(Augereau+ 1999)](https://arxiv.org/abs/astro-ph/9906429) using the **JAX** ecosystem to achieve fast, differentiable, GPU-accelerated computation for disk simulations, parameter inference, and image optimization.

Developed by the **UCSB Exoplanet Polarimetry Lab**, GRaTeR-JAX provides the foundation for analyzing debris disks using modern differentiable programming techniques.

---

## Features
* **GPU/TPU acceleration** through JAX acceleration
* **Differentiable physical modeling** for analytical gradient-based optimization
* **Spline SPF Modeling** for more dynamic and accurate scattering phase function (SPF) fitting
* **PSF convolution** for both static and dynamic PSFs, such as those from _JWST_
* **Modular design** for more flexibility and control
* **Higher Level API** for more intuitive and easy disk fitting
* **Image Processing Utils** for making target, PSF, and error map images

---

## Bugs and Feature Requests
Please use the [GitHub Issue Tracker](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX/issues) to submit bug reports, documentation issues, or feature requests.  
Contributions from the community are always welcome.

---

## Attribution
The development of GRaTeR-JAX is led by **Mihir Kondapalli** and **Briley Lewis**,  
with contributions from members of the **UCSB Exoplanet Polarimetry Lab** and the wider astrophysics software community.

If you build upon this package, please cite both GRaTeR-JAX and the original GRaTeR framework.

---

## Acknowledgments
This work was developed by the **UCSB Exoplanet Polarimetry Lab**.

---

## Contents

```{toctree}
:maxdepth: 2

guide/index
api/index
tutorials/index
