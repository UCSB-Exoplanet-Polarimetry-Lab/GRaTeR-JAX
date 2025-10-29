import numpy as np
import matplotlib.pyplot as plt

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk as VIP_ScatteredLightDisk
from vip_hci.var.filters import frame_filter_lowpass

from grater_jax.optimization.optimize_framework import Optimizer
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk as JAX_ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    HenyeyGreenstein_SPF,
)
from grater_jax.disk_model.objective_functions import Parameter_Index
import jax
import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.10'
jax.config.update("jax_enable_x64", True)


def test_optimizer_generate_model_and_likelihood():
    # VIP baseline model
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
        nx=200,
        ny=200,
        distance=50.0,
        itilt=60.0,
        omega=0.0,
        pxInArcsec=0.01225,
        pa=0.0,
        density_dico=density,
        spf_dico=phase,
        xdo=0.0,
        ydo=0.0,
    )
    vip_img = vip_disk.compute_scattered_light(halfNbSlices=25)
    assert vip_img.shape == (200, 200)

    # Add a small Gaussian PSF blur
    vip_img_psf = frame_filter_lowpass(
        vip_img, mode="gauss", fwhm_size=4.0, conv_mode="convfft"
    )

    # Define Optimizer inputs
    spf_params = HenyeyGreenstein_SPF.params.copy()
    spf_params["g"] = 0.3

    disk_params = Parameter_Index.disk_params.copy()
    misc_params = Parameter_Index.misc_params.copy()

    disk_params.update(
        {
            "sma": 60.0,
            "alpha_in": 5.0,
            "alpha_out": -5.0,
            "ksi0": 1.0,
            "gamma": 2.0,
            "beta": 1.0,
            "e": 0.0,
            "dens_at_r0": 1.0,
            "inclination": 60.0,
            "position_angle": 0.0,
            "omega": 0.0,
            "x_center": 100.0,
            "y_center": 100.0,
            "halfNbSlices": 25,
        }
    )

    misc_params.update(
        {
            "distance": 50.0,
            "pxInArcsec": 0.01225,
            "nx": 200,
            "ny": 200,
            "halfNbSlices": 25,
            "flux_scaling": 1,
        }
    )

    psf_params = {}  # none used for now

    # --- 3. Instantiate Optimizer ---
    optimizer = Optimizer(
        DiskModel=JAX_ScatteredLightDisk,
        DistrModel=DustEllipticalDistribution2PowerLaws,
        FuncModel=HenyeyGreenstein_SPF,
        PSFModel=None,
        disk_params=disk_params,
        spf_params=spf_params,
        psf_params=psf_params,
        misc_params=misc_params,
    )

    # Generate JAX model
    jax_img = optimizer.get_model()
    assert jax_img.shape == (200, 200)
    assert np.isfinite(jax_img).any()

    # Compute likelihood
    err_map = np.ones_like(vip_img_psf) * np.nanstd(vip_img_psf)
    ll = optimizer.log_likelihood(vip_img_psf, err_map)
    assert np.isfinite(ll)
    print("Log-likelihood:", ll)

    # Compare residuals
    residual = vip_img_psf - jax_img
    mean_resid = np.mean(np.abs(residual))
    max_resid = np.max(np.abs(residual))
    print("Mean residual:", mean_resid)
    print("Max residual:", max_resid)

    assert mean_resid < 1e-6
    assert max_resid < 1e-5

    # Visualization
    vmin = min(vip_img_psf.min(), jax_img.min())
    vmax = max(vip_img_psf.max(), jax_img.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(vip_img_psf, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("VIP (PSF-applied)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(jax_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("GRaTeR-JAX (Optimizer)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(residual, cmap="seismic", vmin=-max_resid, vmax=max_resid)
    axes[2].set_title("Residual")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("test_results/test_optimizer_model.png", dpi=200)
    plt.close(fig)
