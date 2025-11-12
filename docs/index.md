# GRaTeR-JAX
**GPU-accelerated modeling of scattered-light disks**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c10f45e8-5449-4891-b6a7-33954cf6d954" width="220">
</p>

**GRaTeR-JAX** is a Python library for forward modeling of scattered light circumstellar debris disks.  
It builds upon the *Generalized Radial Transporter (GRaTeR)* framework using the **JAX** ecosystem to achieve fast, differentiable, GPU-accelerated computation for disk simulations, parameter inference, and image optimization.

Developed by the **UCSB Exoplanet Polarimetry Lab**, GRaTeR-JAX provides the foundation for analyzing debris and protoplanetary disks using modern differentiable programming techniques.

---

## Features
* **GPU/TPU acceleration** through JAX acceleration
* **Differentiable physical modeling** for analytical gradient-based optimization
* **Spline SPF Modeling** for more dynamic and accurate SPF fitting
* **PSF convolution** for both static and dynamic psfs
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
It builds upon open-source scientific computing tools including JAX, NumPy, and Astropy,  
and draws inspiration from related modeling frameworks such as [pyKLIP](https://bitbucket.org/pyklip/pyklip).

---

## Contents

```{toctree}
:maxdepth: 2

api/index
tutorials/index
