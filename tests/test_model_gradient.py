import os
import numpy as np
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.40"

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk as VIP_ScatteredLightDisk
from grater_jax.optimization.optimize_framework import Optimizer, OptimizeUtils
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    InterpolatedUnivariateSpline_SPF,
    EMP_PSF,
)
from grater_jax.disk_model.objective_functions import Parameter_Index

# ================================
# Synthetic PSF
# ================================
def generate_gaussian_psf(size=141, fwhm=5.0):
    sigma = fwhm / 2.355
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf = psf.astype(np.float32)
    psf /= np.sum(psf)
    return psf


# ================================
# Synthetic target from VIP
# ================================
def generate_vip_synthetic_target(nx=200, ny=200):
    density = {
        "name": "2PowerLaws",
        "a": 60.0,
        "ksi0": 1.0,
        "gamma": 2.0,
        "beta": 1.0,
        "ain": 5.0,
        "aout": -5.0,
        "e": 0.0,
        "dens_at_r0": 1.0,
    }
    phase = {"name": "HG", "g": 0.3, "polar": False}

    vip_disk = VIP_ScatteredLightDisk(
        nx=nx, ny=ny,
        distance=50.0,
        itilt=60.0,
        omega=0.0,
        pxInArcsec=0.01225,
        pa=0.0,
        density_dico=density,
        spf_dico=phase,
        xdo=0.0, ydo=0.0
    )

    vip_img = vip_disk.compute_scattered_light(halfNbSlices=25)
    assert vip_img.shape == (nx, ny)
    return vip_img.astype(np.float32)

# ================================
# Gradient test
# ================================
def test_empirical_gradient():
    """Compare analytic and numeric gradients with NO PSF convolution."""

    # Synthetic VIP target
    target_image = generate_vip_synthetic_target(nx=140, ny=140)
    err_map = np.ones_like(target_image, dtype=np.float32)

    # GRaTeR-JAX parameter setup
    start_disk_params = Parameter_Index.disk_params.copy()
    start_spf_params  = InterpolatedUnivariateSpline_SPF.params.copy()

    # No PSF → both the class and params are None
    start_psf_params  = None

    start_misc_params = Parameter_Index.misc_params.copy()

    start_disk_params.update({
        "sma": 46.0,
        "inclination": 80.0,
        "position_angle": 27.5,
        "x_center": 70.0,
        "y_center": 70.0,
        "flux_scaling": 1.0,
    })

    start_spf_params["num_knots"] = 5

    start_misc_params.update({
        "distance": 50.0,
        "nx": 140,
        "ny": 140,
        "pxInArcsec": 0.01225,
    })

    # Initialize optimizer with NO PSF model
    opt = Optimizer(
        ScatteredLightDisk,
        DustEllipticalDistribution2PowerLaws,
        InterpolatedUnivariateSpline_SPF,
        PSFModel=None,
        disk_params=start_disk_params,
        spf_params=start_spf_params,
        psf_params=start_psf_params,
        misc_params=start_misc_params,
    )

    # SPF knots
    opt.inc_bound_knots()
    opt.initialize_knots(target_image)

    # Compile model + gradient (no PSF path)
    opt.jit_compile_model()
    opt.jit_compile_gradient(target_image, err_map)

    # Parameters to test
    fit_keys = ["sma", "alpha_in"]

    analytic_grad = opt.get_objective_gradient(
        [46.0, 5.0], fit_keys, target_image, err_map
    )

    # Finite-difference gradient
    eps = 1e-3
    numeric_grad = np.zeros_like(analytic_grad)
    base_params = opt.get_values(fit_keys)

    for i in range(len(fit_keys)):
        params_up = base_params.copy()
        params_down = base_params.copy()

        params_up[i] += eps
        params_down[i] -= eps

        ll_up = opt.get_objective_likelihood(params_up, fit_keys, target_image, err_map)
        ll_down = opt.get_objective_likelihood(params_down, fit_keys, target_image, err_map)

        numeric_grad[i] = (ll_up - ll_down) / (2 * eps)

    # Diagnostics
    print("\nGradient Comparison (NO PSF):")
    for k, a, n in zip(fit_keys, analytic_grad, numeric_grad):
        print(f"{k:12s} | analytic={a:+.6e}  numeric={n:+.6e}")

    # Checks
    assert np.all(np.isfinite(analytic_grad))
    assert np.all(np.isfinite(numeric_grad))
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-2, atol=1e-4)

# ================================
# Gradient test (with PSF)
# ================================
def test_empirical_gradient_PSF():
    """Compare analytic and numeric gradients on synthetic VIP disk + synthetic PSF."""

    # Synthetic inputs
    emp_psf_image = generate_gaussian_psf(size=141, fwhm=5.0)
    target_image = generate_vip_synthetic_target(nx=140, ny=140)
    err_map = np.ones_like(target_image, dtype=np.float32)

    # GRaTeR-JAX parameter setup
    start_disk_params = Parameter_Index.disk_params.copy()
    start_spf_params  = InterpolatedUnivariateSpline_SPF.params.copy()
    start_psf_params  = EMP_PSF.params.copy()
    start_misc_params = Parameter_Index.misc_params.copy()

    start_disk_params.update({
        "sma": 46.0,
        "inclination": 80.0,
        "position_angle": 27.5,
        "x_center": 70.0,
        "y_center": 70.0,
        "flux_scaling": 1.0,
    })

    start_spf_params["num_knots"] = 5

    # IMPORTANT: distance must match the VIP synthetic generator
    start_misc_params.update({
        "distance": 50.0,
        "nx": 140,
        "ny": 140,
        "pxInArcsec": 0.01225,
    })

    # Initialize optimizer
    opt = Optimizer(
        ScatteredLightDisk,
        DustEllipticalDistribution2PowerLaws,
        InterpolatedUnivariateSpline_SPF,
        EMP_PSF,
        start_disk_params,
        start_spf_params,
        start_psf_params,
        start_misc_params,
    )

    opt.inc_bound_knots()
    opt.set_empirical_psf(emp_psf_image)
    opt.initialize_knots(target_image)

    opt.jit_compile_model()
    opt.jit_compile_gradient(target_image, err_map)

    # Parameters to check
    fit_keys = ["sma", "alpha_in"]

    analytic_grad = opt.get_objective_gradient(
        [46.0, 5.0], fit_keys, target_image, err_map
    )

    # Numeric finite differences
    eps = 1e-3
    numeric_grad = np.zeros_like(analytic_grad)
    base_params = opt.get_values(fit_keys)

    for i in range(len(fit_keys)):
        params_up = base_params.copy()
        params_down = base_params.copy()

        params_up[i] += eps
        params_down[i] -= eps

        ll_up = opt.get_objective_likelihood(params_up, fit_keys, target_image, err_map)
        ll_down = opt.get_objective_likelihood(params_down, fit_keys, target_image, err_map)

        numeric_grad[i] = (ll_up - ll_down) / (2 * eps)

    # Diagnostics
    print("\nGradient Comparison:")
    for k, a, n in zip(fit_keys, analytic_grad, numeric_grad):
        print(f"{k:12s} | analytic={a:+.6e}  numeric={n:+.6e}")

    # Assertions
    assert np.all(np.isfinite(analytic_grad))
    assert np.all(np.isfinite(numeric_grad))
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-2, atol=1e-4)
