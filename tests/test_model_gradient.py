import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import jax
jax.config.update("jax_enable_x64", True)
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.40'
os.environ["WEBBPSF_PATH"] = 'stpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'stpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"

from grater_jax.optimization.optimize_framework import Optimizer, OptimizeUtils
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    InterpolatedUnivariateSpline_SPF,
    EMP_PSF,
)
from grater_jax.disk_model.objective_functions import Parameter_Index
import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.60'

def load_empirical_psf():
    """Load and crop the empirical PSF (GPI H-band), normalize to sum=1."""
    image = fits.open("test_images/GPI_Hband_PSF.fits")[0].data[0, :, :]
    cropped = np.nan_to_num(image[70:211, 70:211]).astype(np.float32)
    cropped /= np.sum(cropped)
    return cropped


def test_empirical_psf_hd115600_gradient():
    """Compare analytic and numeric gradients."""

    # Load target FITS images (science + error)
    fits_image_filepath = "test_images/hd115600_H_pol.fits"
    hdul = fits.open(fits_image_filepath)
    target_image = OptimizeUtils.process_image(
        hdul["SCI"].data[1, :, :], bounds=(40, 240, 40, 240)
    )
    err_map = OptimizeUtils.process_image(
        OptimizeUtils.create_empirical_err_map(hdul["SCI"].data[2, :, :]),
        bounds=(40, 240, 40, 240),
    )

    # Loading empirical PSF
    emp_psf_image = load_empirical_psf()

    # Defining parameters
    row = {
        "Name": "hd115600_H_pol",
        "Radius": 46.0,
        "Inclination": 80.0,
        "Position Angle": 27.5,
        "Distance": 109.62,
        "Knots": 5,
    }

    start_disk_params = Parameter_Index.disk_params.copy()
    start_spf_params = InterpolatedUnivariateSpline_SPF.params.copy()
    start_psf_params = EMP_PSF.params.copy()
    start_misc_params = Parameter_Index.misc_params.copy()

    start_disk_params.update(
        {
            "sma": row["Radius"],
            "inclination": row["Inclination"],
            "position_angle": row["Position Angle"],
            "x_center": 100.0,
            "y_center": 100.0,
        }
    )

    start_spf_params["num_knots"] = int(row["Knots"])

    start_misc_params.update(
        {
            "distance": row["Distance"],
            "nx": 200,
            "ny": 200,
        }
    )

    # Initialize optimizer with spline SPF + empirical PSF
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

    opt.inc_bound_knots()  # optional bound setup
    opt.set_empirical_psf(emp_psf_image)
    opt.initialize_knots(target_image)

    opt.jit_compile_model()
    opt.jit_compile_gradient(target_image, err_map)

    fit_keys = ["sma", "alpha_in"]
    analytic_grad = opt.get_objective_gradient([46.0, 5.0], fit_keys, target_image, err_map)

    eps = 1e-3
    numeric_grad = np.zeros_like(analytic_grad)

    base_params = opt.get_values(fit_keys)
    for i, key in enumerate(fit_keys):
        params_up = base_params.copy()
        params_down = base_params.copy()

        params_up[i] += eps
        params_down[i] -= eps

        ll_up = opt.get_objective_likelihood(params_up, fit_keys, target_image, err_map)
        ll_down = opt.get_objective_likelihood(params_down, fit_keys, target_image, err_map)
        numeric_grad[i] = (ll_up - ll_down) / (2 * eps)

    print("\nGradient Comparison (HD115600 + Empirical PSF):")
    for k, a, n in zip(fit_keys, analytic_grad, numeric_grad):
        diff = abs(a - n)
        rel_err = diff / (abs(n) + 1e-12)
        print(f"{k:12s} | analytic={a:+.6e}  numeric={n:+.6e}  diff={diff:.3e}  rel_err={rel_err:.3e}")

    # Check gradients are finite and consistent
    assert np.all(np.isfinite(analytic_grad)), "Analytic gradient has NaNs/Infs"
    assert np.all(np.isfinite(numeric_grad)), "Numeric gradient has NaNs/Infs"
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-2, atol=1e-4), \
        "Analytic and numeric gradients differ beyond tolerance"
