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
* **GPU/TPU acceleration** through JAX
* **Differentiable physical modeling** for analytical gradient-based optimization
* **Spline SPF Modeling** for more dynamic and accurate scattering phase function (SPF) fitting
* **PSF convolution** for both static and dynamic PSFs, such as those from _JWST_
* **Modular design** for more flexibility and control
* **Higher Level API** for more intuitive and user-friendly disk fitting
* **Image Processing Utilities** for ingesting data, making PSF models, and estimating noise

---

## Bugs and Feature Requests
Please use the [GitHub Issue Tracker](https://github.com/UCSB-Exoplanet-Polarimetry-Lab/GRaTeR-JAX/issues) to submit bug reports, documentation issues, or feature requests.
Contributions from the community are always welcome.

---

## Attribution
The development of GRaTeR-JAX is led by **Mihir Kondapalli**, **Briley Lewis**, **Jaren Ashcraft**, and **Max Millar-Blanchaer**
with contributions from members of the **UCSB Exoplanet Polarimetry Lab** and the wider astrophysics software community.

If you build upon this package, please cite both GRaTeR-JAX (Kondapalli et al. in prep JOSS, Lewis et al. in prep AJ) and the original GRaTeR framework [(Augereau+ 1999)](https://ui.adsabs.harvard.edu/abs/1999A%26A...348..557A/abstract).

---

## Acknowledgments
This work was developed by the [**UCSB Exoplanet Polarimetry Lab**](https://ucsb-exoplanet-polarimetry-lab.github.io/).

---

## Contents

```{toctree}
:maxdepth: 2

index
api/index
Installation/index
Tutorials/index
How-Tos/index
Explanations/index
```
