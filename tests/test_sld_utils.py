# tests/test_utils_and_models.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import jax.scipy.signal as jss
import matplotlib.pyplot as plt
import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.10'
jax.config.update("jax_enable_x64", True)

from grater_jax.disk_model.SLD_utils import (
    Jax_class,
    DustEllipticalDistribution2PowerLaws,
    HenyeyGreenstein_SPF,
    DoubleHenyeyGreenstein_SPF,
    GAUSSIAN_PSF,
    LinearStellarPSF,
    PositionalStellarPSF,
    StellarPSFReference,
)

# Function mocks

def ref_hg(cos_phi, g):
    """Closed-form single Henyey–Greenstein phase function."""
    g = jnp.clip(g, -0.999, 0.999)
    return (1.0 / (4.0 * jnp.pi)) * (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * cos_phi) ** 1.5

def ref_double_hg(cos_phi, g1, g2, weight):
    """Closed-form double Henyey–Greenstein (convex combo of two HGs)."""
    return weight * ref_hg(cos_phi, g1) + (1.0 - weight) * ref_hg(cos_phi, g2)

def ref_density_Dust2PL(distr_params, r, costheta, z):
    """Reference computation matching DustEllipticalDistribution2PowerLaws.density_cylindrical."""
    d = DustEllipticalDistribution2PowerLaws.unpack_pars(distr_params)

    # Radial ratio & terms
    radial_ratio = r * (1.0 - d["e"] * costheta) / (d["p"] + 1e-8)
    rr = jnp.abs(radial_ratio) + 1e-8
    den = rr ** (-2.0 * d["alpha_in"]) + rr ** (-2.0 * d["alpha_out"])
    radial_density_term = jnp.sqrt(2.0 / (den + 1e-8)) * d["dens_at_r0"]

    # Inner hole gating (elliptical)
    radial_density_term = jnp.where(
        d["pmin"] > 0,
        jnp.where(r * (1.0 - d["e"] * costheta) / (d["p"] + 1e-8) <= 1.0, 0.0, radial_density_term),
        radial_density_term,
    )

    # Vertical term
    den2 = d["ksi0"] * (jnp.abs(radial_ratio + 1e-8) ** d["beta"]) + 1e-8
    vertical_density_term = jnp.exp(
        -((jnp.abs(z) + 1e-8) / (jnp.abs(den2 + 1e-8))) ** (jnp.abs(d["gamma"]) + 1e-8)
    )

    return radial_density_term * vertical_density_term

def ref_gaussian_kernel(nx, ny, FWHM, xo, yo, theta, offset, amplitude):
    """Reference isotropic Gaussian kernel (matches current GAUSSIAN_PSF math: b == 0)."""
    # NOTE: SLD_utils computes a == c == 1/(2*sigma^2) and b == 0, so theta cancels out.
    sigma = FWHM / 2.355
    y = jnp.linspace(-ny // 2, ny // 2, ny)
    x = jnp.linspace(-nx // 2, nx // 2, nx)
    X, Y = jnp.meshgrid(x, y)  # (ny, nx)
    R2 = (X - xo) ** 2 + (Y - yo) ** 2
    return offset + amplitude * jnp.exp(-R2 / (2.0 * sigma**2))

def ref_gaussian_convolution(image, FWHM, xo, yo, theta, offset, amplitude):
    ker = ref_gaussian_kernel(image.shape[1], image.shape[0], FWHM, xo, yo, theta, offset, amplitude)
    return jss.convolve2d(image, ker, mode="same")

# Helpers

def _trapz_1d(y, x):
    """Pure-jnp trapezoid rule: sum 0.5*(y[i]+y[i-1]) * (x[i]-x[i-1])."""
    dx = x[1:] - x[:-1]
    return jnp.sum(0.5 * (y[1:] + y[:-1]) * dx)

def trapz_2pi_over_cos(vals, cos_grid):
    """Integrate phase function over the sphere: 2π ∫_{-1}^{1} f(c) dc."""
    return 2.0 * jnp.pi * _trapz_1d(vals, cos_grid)

def make_impulse(h, w, yx):
    img = jnp.zeros((h, w))
    return img.at[yx[0], yx[1]].set(1.0)

# Jax_class

def test_pack_unpack_roundtrip_and_order():
    class Dummy(Jax_class):
        # Intentional non-lexical order to verify packing order is respected
        params = {"b": 2.0, "a": 1.0}

    arr = jnp.array([3.0, 4.0])  # maps to b=3, a=4 in this class
    d = Dummy.unpack_pars(arr)
    assert d["b"] == pytest.approx(3.0)
    assert d["a"] == pytest.approx(4.0)

    arr2 = Dummy.pack_pars(d)
    # Must come back in the class's declared order ("b", then "a")
    assert jnp.allclose(arr2, jnp.array([3.0, 4.0]))

# DustEllipticalDistribution2PowerLaws

def test_density_matches_reference_selected_points():
    pars = DustEllipticalDistribution2PowerLaws.init(
        alpha_in=5.0, alpha_out=-5.0, sma=60.0, e=0.0, ksi0=1.0, gamma=2.0, beta=1.0,
        rmin=0.0, dens_at_r0=1.0
    )
    samples = [
        (30.0,  0.0, 0.0),
        (60.0,  0.0, 0.0),
        (90.0,  0.0, 0.0),
        (60.0,  0.5, 0.0),
        (60.0, -0.8, 1.0),
    ]
    for r, costh, z in samples:
        got = DustEllipticalDistribution2PowerLaws.density_cylindrical(pars, r, costh, z)
        ref = ref_density_Dust2PL(pars, r, costh, z)
        assert got == pytest.approx(float(ref), rel=1e-6, abs=1e-8)

def test_density_decreases_away_from_midplane():
    pars = DustEllipticalDistribution2PowerLaws.init(
        alpha_in=5.0, alpha_out=-5.0, sma=60.0, e=0.0, ksi0=1.0, gamma=2.0, beta=1.0,
        rmin=0.0, dens_at_r0=1.0
    )
    r = 60.0
    cos_theta = 0.0
    zs = jnp.array([0.0, 1.0, 2.0, 3.0])
    rhos = jnp.array([DustEllipticalDistribution2PowerLaws.density_cylindrical(pars, r, cos_theta, z) for z in zs])
    assert jnp.all(rhos[:-1] >= rhos[1:])
    assert rhos[0] > rhos[-1]

# Henyey-Greenstein SPF

def test_hg_matches_reference_function():
    g = 0.5
    pars = HenyeyGreenstein_SPF.init(jnp.array([g]))
    cos_grid = jnp.linspace(-1.0, 1.0, 5001)
    got = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, cos_grid)
    ref = ref_hg(cos_grid, g)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)

def test_hg_phase_function_normalization():
    g = 0.5
    pars = HenyeyGreenstein_SPF.init(jnp.array([g]))
    cos_grid = jnp.linspace(-1.0, 1.0, 20_001)
    vals = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, cos_grid)
    integral = trapz_2pi_over_cos(vals, cos_grid)
    assert integral == pytest.approx(1.0, rel=1e-3, abs=1e-3)

@pytest.mark.parametrize("g, expect_forward_higher", [(0.5, True), (-0.5, False)])
def test_hg_forward_vs_backward_scattering(g, expect_forward_higher):
    pars = HenyeyGreenstein_SPF.init(jnp.array([g]))
    fwd = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, jnp.array([1.0]))[0]
    back = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, jnp.array([-1.0]))[0]
    if expect_forward_higher:
        assert fwd > back
    else:
        assert back > fwd

# Double Henyey-Greenstein SPF

def test_double_hg_matches_reference_function():
    g1, g2, w = 0.5, -0.3, 0.7
    pars = DoubleHenyeyGreenstein_SPF.init(jnp.array([g1, g2, w]))
    cos_grid = jnp.linspace(-1.0, 1.0, 5001)
    got = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, cos_grid)
    ref = ref_double_hg(cos_grid, g1, g2, w)
    assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)

def test_double_hg_normalization():
    pars = DoubleHenyeyGreenstein_SPF.init(jnp.array([0.5, -0.3, 0.7]))
    cos_grid = jnp.linspace(-1.0, 1.0, 20_001)
    vals = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars, cos_grid)
    integral = trapz_2pi_over_cos(vals, cos_grid)
    assert integral == pytest.approx(1.0, rel=1e-3, abs=1e-3)

def test_double_hg_reduces_to_single_when_weight_1():
    g = 0.6
    pars_double = DoubleHenyeyGreenstein_SPF.init(jnp.array([g, -0.2, 1.0]))
    pars_single = HenyeyGreenstein_SPF.init(jnp.array([g]))
    cos_grid = jnp.linspace(-1.0, 1.0, 1001)
    vals_double = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars_double, cos_grid)
    vals_single = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(pars_single, cos_grid)
    assert jnp.allclose(vals_double, vals_single, rtol=1e-6, atol=1e-8)

# Gaussian PSF

def _gaussian_psf_generate(image, FWHM=2.0, xo=0.0, yo=0.0, theta=0.0, offset=0.0, amplitude=1.0):
    psf_pars = GAUSSIAN_PSF.pack_pars(
        {"FWHM": FWHM, "xo": xo, "yo": yo, "theta": theta, "offset": offset, "amplitude": amplitude}
    )
    return GAUSSIAN_PSF.generate(image, psf_pars)

def test_gaussian_psf_matches_reference_on_impulse():
    """
    Verifies GAUSSIAN_PSF.generate against the minimal Fourier-domain reference.
    Because GAUSSIAN_PSF implements an isotropic Fourier Gaussian, the reference
    must match exactly that math.
    """
    H = W = 33
    impulse = make_impulse(H, W, (H // 2, W // 2))

    def ref_fft_gaussian(image, FWHM, amplitude, offset):
        sigma = FWHM / 2.355
        ny, nx = image.shape
        fx = jnp.fft.fftfreq(nx)
        fy = jnp.fft.fftfreq(ny)
        FX, FY = jnp.meshgrid(fx, fy)
        G = amplitude * jnp.exp(-(2 * jnp.pi**2) * sigma**2 * (FX**2 + FY**2))
        return jnp.fft.ifft2(jnp.fft.fft2(image) * G).real + offset

    for FWHM in [1.5, 2.0, 4.0]:
        for amp in [0.5, 1.0]:
            for off in [0.0, 0.1]:
                got = _gaussian_psf_generate(
                    impulse,
                    FWHM=FWHM,
                    xo=0.0,
                    yo=0.0,
                    theta=0.0,   # ignored
                    offset=off,
                    amplitude=amp
                )
                ref = ref_fft_gaussian(impulse, FWHM, amp, off)
                assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)

def test_gaussian_psf_linearity_and_shift_invariance():
    H = W = 33
    i1 = make_impulse(H, W, (H // 2, W // 2 - 5))
    i2 = make_impulse(H, W, (H // 2 - 4, W // 2 + 3))
    both = i1 + i2

    out1 = _gaussian_psf_generate(i1, FWHM=2.0)
    out2 = _gaussian_psf_generate(i2, FWHM=2.0)
    out_both = _gaussian_psf_generate(both, FWHM=2.0)

    assert jnp.allclose(out_both, out1 + out2, rtol=1e-6, atol=1e-8)

# -------------------------
# Stellar PSFs
# -------------------------

def test_linear_stellar_psf_vs_reference_resize_combo():
    # Reference: tensordot + jax.image.resize (linear)
    old_refs = StellarPSFReference.reference_images
    try:
        ref1 = jnp.ones((5, 5))
        ref2 = 2.0 * jnp.ones((5, 5))
        StellarPSFReference.reference_images = jnp.stack([ref1, ref2], axis=0)  # (2,5,5)

        weights = jnp.array([1.0, 0.5])
        nx, ny = 10, 8

        # GRaTeR-JAX
        got = LinearStellarPSF.compute_stellar_psf_image(weights, nx=nx, ny=ny)

        # Reference
        combo = jnp.tensordot(weights, StellarPSFReference.reference_images, axes=1)  # (5,5)
        ref = jax.image.resize(combo, (nx, ny), method="linear")

        assert got.shape == (nx, ny)
        assert jnp.allclose(got, ref, rtol=1e-6, atol=1e-8)
    finally:
        StellarPSFReference.reference_images = old_refs

from grater_jax.disk_model.SLD_utils import PositionalStellarPSF, StellarPSFReference

def _ref_stitch_positional(refs, weights, xs, ys, nx, ny):
    """
    Reference stitcher that mirrors PositionalStellarPSF's bilinear placement,
    masking out-of-bounds and summing overlapping contributions.
    refs: (N, h, w), weights/xs/ys: (N,), canvas shape (nx, ny)
    """
    refs = jnp.asarray(refs)
    weights = jnp.asarray(weights)
    xs = jnp.asarray(xs)
    ys = jnp.asarray(ys)

    N, h, w = refs.shape
    xx = jnp.arange(h).reshape(h, 1)  # (h,1)
    yy = jnp.arange(w).reshape(1, w)  # (1,w)

    acc = jnp.zeros((nx, ny), dtype=refs.dtype)

    for i in range(N):
        psf_img = refs[i]                # (h, w)
        wt = weights[i]
        x = xs[i]
        y = ys[i]

        x0 = x - h / 2.0
        y0 = y - w / 2.0

        x_pix = x0 + xx                  # (h,1)
        y_pix = y0 + yy                  # (1,w)

        x0f = jnp.floor(x_pix)           # (h,1)
        y0f = jnp.floor(y_pix)           # (1,w)

        dx = (x_pix - x0f).repeat(w, axis=1)   # (h,w)
        dy = (y_pix - y0f).repeat(h, axis=0)   # (h,w)

        x0i = x0f.astype(jnp.int32).repeat(w, axis=1)  # (h,w)
        y0i = y0f.astype(jnp.int32).repeat(h, axis=0)  # (h,w)

        # Bilinear weights
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        # For each of the 4 neighbors, accumulate values within bounds
        for (sx, sy, wm) in [(0, 0, w00), (1, 0, w10), (0, 1, w01), (1, 1, w11)]:
            xi = (x0i + sx).reshape(-1)
            yi = (y0i + sy).reshape(-1)
            val = (wt * psf_img * wm).reshape(-1)

            mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
            xi = xi[mask]
            yi = yi[mask]
            val = val[mask]

            acc = acc.at[xi, yi].add(val)

    return acc


def test_positional_stellar_psf_matches_reference_stitch_multiple_refs():
    """
    Define a small bank of reference images and place them at various (x,y)
    with weights; verify PositionalStellarPSF reproduces the exact stitched canvas.
    """
    old_refs = StellarPSFReference.reference_images
    try:
        # Three distinct 3x3 reference images (deterministic patterns)
        ref0 = jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3)          # 0..8
        ref1 = (jnp.arange(9.0, dtype=jnp.float32).reshape(3, 3))[::-1] + 10.0  # flipped + offset
        ref2 = jnp.array([[0., 1., 0.],
                          [1., 2., 1.],
                          [0., 1., 0.]], dtype=jnp.float32) * 3.0

        refs = jnp.stack([ref0, ref1, ref2], axis=0)  # (N=3, h=3, w=3)
        StellarPSFReference.reference_images = refs

        # Canvas and placements (kept away from edges so nothing is clipped)
        nx, ny = 12, 10
        weights = jnp.array([0.7, 1.0, 0.5], dtype=jnp.float32)
        xs = jnp.array([3.7, 6.25, 8.1], dtype=jnp.float32)
        ys = jnp.array([4.4, 3.2, 6.9], dtype=jnp.float32)

        # Reference stitch
        expected = _ref_stitch_positional(refs, weights, xs, ys, nx=nx, ny=ny)

        # PositionalStellarPSF
        params = jnp.concatenate([weights, xs, ys])  # pack: [weights, xs, ys]
        got = PositionalStellarPSF.compute_stellar_psf_image(params, nx=nx, ny=ny)

        # --- Residual (VIP PSF vs JAX no-PSF) ---
        residual = got - expected

        # --- Plot with shared scale for VIP & JAX; residual uses same scale as VIP/JAX per your request ---
        vmin = min(got.min(), expected.min())
        vmax = max(got.max(), expected.max())

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im0 = axes[0].imshow(got, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0].set_title("Expected Stellar PSF")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(expected, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1].set_title("GRaTeR-JAX Generated Stellar PSF")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Residual shown on the same 'inferno' scale to compare brightness directly
        im2 = axes[2].imshow(residual, cmap="inferno", vmin=vmin, vmax=vmax)
        axes[2].set_title("Residual")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("test_results/test_positional_stellar_psf.png", dpi=200)
        plt.close(fig)

        assert got.shape == (nx, ny)
        assert jnp.allclose(got, expected, rtol=1e-6, atol=1e-8)

        # Mass conservation (no clipping): sum should match weighted sums of refs
        total_ref_mass = jnp.sum(weights[0] * jnp.sum(ref0) +
                                 weights[1] * jnp.sum(ref1) +
                                 weights[2] * jnp.sum(ref2))
        assert jnp.sum(got) == pytest.approx(float(total_ref_mass), rel=1e-6, abs=1e-6)
    finally:
        StellarPSFReference.reference_images = old_refs

def test_gaussian_psf_shape_only():
    image = jnp.zeros((32, 32))
    pars = GAUSSIAN_PSF.pack_pars(
        {"FWHM": 2.0, "xo": 0.0, "yo": 0.0, "theta": 0.0, "offset": 0.0, "amplitude": 1.0}
    )
    out = GAUSSIAN_PSF.generate(image, pars)
    assert jnp.shape(out) == (32, 32)
