import numpy as np

from vip_hci.fm.scattered_light_disk import ScatteredLightDisk as VIP_ScatteredLightDisk
from vip_hci.var.filters import frame_filter_lowpass  # <-- VIP gaussian PSF
from grater_jax.disk_model.objective_functions import objective_model, Parameter_Index
from grater_jax.disk_model.SLD_utils import DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF, GAUSSIAN_PSF
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk as JAX_ScatteredLightDisk
import matplotlib.pyplot as plt
import jax
import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.60'
jax.config.update("jax_enable_x64", True)


def test_simple_vip_and_grater_jax_generate_images():
    # VIP disk
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
        nx=200, ny=200,
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
    assert isinstance(vip_img, np.ndarray)
    assert vip_img.shape == (200, 200)

    # --- GRaTeR-JAX disk ---
    spf_params = HenyeyGreenstein_SPF.params
    disk_params = Parameter_Index.disk_params.copy()
    misc_params = Parameter_Index.disk_params.copy()

    disk_params.update({
        "sma": 60.0,                # matches density["a"]
        "alpha_in": 5.0,            # density["ain"]
        "alpha_out": -5.0,          # density["aout"]
        "ksi0": 1.0,                # density["ksi0"]
        "gamma": 2.0,               # density["gamma"]
        "beta": 1.0,                # density["beta"]
        "e": 0.0,                   # density["e"]
        "dens_at_r0": 1.0,          # density["dens_at_r0"]

        "inclination": 60.0,        # itilt
        "position_angle": 0.0,      # pa
        "omega": 0.0,               # omega
        "x_center": 100.0,          # nx/2 = 200/2
        "y_center": 100.0,          # ny/2 = 200/2
        "halfNbSlices": 25,         # compute_scattered_light arg
    })

    misc_params.update({
        "distance": 50.0,           # distance
        "pxInArcsec": 0.01225,      # pxInArcsec
        "nx": 200,                  # nx
        "ny": 200,                  # ny
        "halfNbSlices": 25,         # consistency with disk_params
        "flux_scaling": 1,          # keep default
    })

    print(misc_params['nx'])


    jax_img = objective_model(
        disk_params, spf_params, None, misc_params,
        JAX_ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
        HenyeyGreenstein_SPF, None
    )

    assert jax_img.shape == (200, 200)

    # Basic sanity comparison
    assert np.isfinite(vip_img).any()
    assert np.isfinite(jax_img).any()

    # Residual
    residual = vip_img - jax_img

    # Assert residual is negligible (tolerance can be tuned)
    max_resid = np.max(np.abs(residual))
    mean_resid = np.mean(np.abs(residual))

    print("Max residual:", max_resid)
    print("Mean residual:", mean_resid)

    # Allow for floating point + implementation differences
    assert max_resid < 1e-6, f"Residual too large: {max_resid}"
    assert mean_resid < 1e-7, f"Mean residual too large: {mean_resid}"

    # VIP vs GRaTeR-JAX vs Residuals
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute global limits
    vmin = min(vip_img.min(), jax_img.min())
    vmax = max(vip_img.max(), jax_img.max())

    # For residual, use symmetric around 0
    res_limit = np.max(np.abs(residual))

    # --- Plot with shared limits ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute global limits
    vmin = min(vip_img.min(), jax_img.min())
    vmax = max(vip_img.max(), jax_img.max())

    # For residual, use symmetric around 0
    res_limit = np.max(np.abs(residual))

    # --- Plot with shared limits ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Compute global limits
    vmin = min(vip_img.min(), jax_img.min())
    vmax = max(vip_img.max(), jax_img.max())

    # For residual, use symmetric around 0
    res_limit = np.max(np.abs(residual))

    # Plotting on same scale
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(vip_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("VIP Image")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(jax_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("GRaTeR-JAX Image")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(residual, cmap="seismic", vmin=-res_limit, vmax=res_limit)
    axes[2].set_title("Residual (VIP - GRaTeR-JAX)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("test_results/simple_vip_grater_model_comparison.png", dpi=200)
    plt.close(fig)


def test_gaussian_psf_vip_and_grater_jax_generate_images():
    # VIP disk
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
        nx=200, ny=200,
        distance=50.0,
        itilt=60.0,
        omega=0.0,
        pxInArcsec=0.01225,
        pa=0.0,
        density_dico=density,
        spf_dico=phase,
        xdo=0.0, ydo=0.0
    )
    vip_img_raw = vip_disk.compute_scattered_light(halfNbSlices=25)
    assert isinstance(vip_img_raw, np.ndarray)
    assert vip_img_raw.shape == (200, 200)

    # VIP Disk + Gaussian PSF
    fwhm_px = 5.0
    vip_img = frame_filter_lowpass(
        vip_img_raw, mode="gauss", fwhm_size=fwhm_px, conv_mode="convfft"
    )
    assert vip_img.shape == (200, 200)

    # GRaTeR-JAX disk w/ PSF
    spf_params = HenyeyGreenstein_SPF.params.copy()
    spf_params["g"] = 0.3

    psf_params = GAUSSIAN_PSF.params.copy()
    psf_params.update({'FWHM': fwhm_px, 'xo': 0., 'yo': 0., 'theta': 0., 'offset': 0., 'amplitude': 1.})

    disk_params = Parameter_Index.disk_params.copy()
    misc_params = Parameter_Index.misc_params.copy()

    disk_params.update({
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
    })

    misc_params.update({
        "distance": 50.0,
        "pxInArcsec": 0.01225,
        "nx": 200,
        "ny": 200,
        "halfNbSlices": 25,
        "flux_scaling": 1,
    })

    jax_img = objective_model(
        disk_params, spf_params, psf_params, misc_params,
        JAX_ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
        HenyeyGreenstein_SPF, GAUSSIAN_PSF
    )

    assert jax_img.shape == (200, 200)

    # Sanity checks
    assert np.isfinite(vip_img).any()
    assert np.isfinite(jax_img).any()

    # Residual (VIP PSF vs JAX no-PSF)
    residual = vip_img - jax_img
    print("Residual stats (PSF VIP vs PSF JAX):",
          "min=", residual.min(), "max=", residual.max(), "mean|.|=", np.mean(np.abs(residual)))

    assert np.float64(np.mean(np.abs(residual))) < 1e-7

    # Plot with shared scale for VIP & JAX
    vmin = min(vip_img.min(), jax_img.min())
    vmax = max(vip_img.max(), jax_img.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(vip_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("VIP (Gaussian PSF, FWHM=4 px)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(jax_img, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("GRaTeR-JAX")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Residual plot
    im2 = axes[2].imshow(residual, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[2].set_title("Residual (VIP-PSF minus JAX)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("test_results/psf_vip_grater_model_comparison.png", dpi=200)
    plt.close(fig)