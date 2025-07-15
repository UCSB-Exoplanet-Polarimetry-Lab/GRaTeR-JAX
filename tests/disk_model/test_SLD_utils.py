import jax
import jax.numpy as jnp
import numpy as np
import pytest

import os
os.environ["WEBBPSF_PATH"] = 'stpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'stpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"

from disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws,
    HenyeyGreenstein_SPF,
    DoubleHenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF,
    GAUSSIAN_PSF,
)

def test_pack_unpack_consistency():
    init_arr = jnp.array([5., -5., 60., 0., 1., 2., 1., 0., 1., 0.005])
    packed = DustEllipticalDistribution2PowerLaws.init(*init_arr)
    unpacked = DustEllipticalDistribution2PowerLaws.unpack_pars(packed)
    assert np.isclose(unpacked["alpha_in"], 5.)
    assert np.isclose(unpacked["alpha_out"], -5.)
    assert "zmax" in unpacked


def test_density_output_valid():
    p = DustEllipticalDistribution2PowerLaws.init(*jnp.array([5., -5., 60., 0., 1., 2., 1., 0., 1., 0.005]))
    r = jnp.ones(10)
    theta = jnp.linspace(-1, 1, 10)
    z = jnp.linspace(-1, 1, 10)
    density = DustEllipticalDistribution2PowerLaws.density_cylindrical(p, r, theta, z)
    assert density.shape == (10,)
    assert jnp.all(jnp.isfinite(density))
    assert jnp.all(density >= 0)


def test_density_gradient_finite():
    def wrapper(p_arr):
        return DustEllipticalDistribution2PowerLaws.density_cylindrical(
            DustEllipticalDistribution2PowerLaws.init(*p_arr),
            1.0, 0.5, 0.1
        )
    grad_fn = jax.grad(lambda p: jnp.sum(wrapper(p)))
    grad = grad_fn(jnp.array([5., -5., 60., 0., 1., 2., 1., 0., 1., 0.005]))
    assert jnp.all(jnp.isfinite(grad))


def test_hg_phase_function_bounds():
    p = HenyeyGreenstein_SPF.init(jnp.array([0.5]))
    cos_phi = jnp.linspace(-1, 1, 50)
    pf = HenyeyGreenstein_SPF.compute_phase_function_from_cosphi(p, cos_phi)
    assert pf.shape == (50,)
    assert jnp.all(pf >= 0)
    assert jnp.all(jnp.isfinite(pf))


def test_dhg_phase_function_valid():
    p = DoubleHenyeyGreenstein_SPF.init(jnp.array([0.5, -0.3, 0.7]))
    cos_phi = jnp.linspace(-1, 1, 50)
    pf = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(p, cos_phi)
    assert pf.shape == (50,)
    assert jnp.all(pf >= 0)
    assert jnp.all(jnp.isfinite(pf))


def test_interpolated_spline_output():
    knot_values = jnp.array([1., 0.9, 0.8, 0.7, 0.6, 0.5])
    model = InterpolatedUnivariateSpline_SPF.init(knot_values)
    cos_phi = jnp.linspace(-1, 1, 20)
    result = InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(model, cos_phi)
    assert result.shape == (20,)
    assert jnp.all(jnp.isfinite(result))


def test_gaussian_psf_output():
    image = jnp.ones((50, 50))
    psf_params = jnp.array([3., 0., 0., 0., 0., 1.])  # Basic centered PSF
    result = GAUSSIAN_PSF.generate(image, psf_params)
    assert result.shape == (50, 50)
    assert jnp.all(jnp.isfinite(result))
