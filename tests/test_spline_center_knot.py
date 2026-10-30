import numpy as np
import jax
import jax.numpy as jnp
import pytest

from grater_jax.disk_model.objective_functions import objective_model, Parameter_Index
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF, InterpolatedUnivariateSpline_SPF,
)
from grater_jax.optimization.optimize_framework import Optimizer

jax.config.update("jax_enable_x64", True)

G = 0.3


def _hg(cos_phi, g=G):
    return (1.0 / (4.0 * np.pi)) * (1 - g**2) / (1 + g**2 - 2 * g * cos_phi) ** 1.5


@pytest.mark.parametrize("num_knots", [4, 5, 6, 7, 20])
def test_center_index_matches_get_knots_zero_position(num_knots):
    spf = InterpolatedUnivariateSpline_SPF.params.copy()
    spf["num_knots"] = num_knots
    knots = np.array(InterpolatedUnivariateSpline_SPF.get_knots(spf))
    ci = InterpolatedUnivariateSpline_SPF.center_index(num_knots)
    assert knots[ci] == 0.0
    assert len(knots) == num_knots + 1


@pytest.mark.parametrize("num_knots", [5, 7])
def test_packed_spline_puts_unit_value_at_the_zero_knot(num_knots):
    spf = InterpolatedUnivariateSpline_SPF.params.copy()
    spf["num_knots"] = num_knots
    knots = InterpolatedUnivariateSpline_SPF.get_knots(spf)
    # Distinct free values so a misplaced centre is visible
    free = jnp.arange(2.0, 2.0 + num_knots)
    spline = InterpolatedUnivariateSpline_SPF.pack_pars(free, knots)
    ci = InterpolatedUnivariateSpline_SPF.center_index(num_knots)
    np.testing.assert_allclose(np.array(spline._y)[ci], 1.0)
    expected = np.concatenate([np.array(free)[:ci], [1.0], np.array(free)[ci:]])
    np.testing.assert_allclose(np.array(spline._y), expected)


@pytest.mark.parametrize("num_knots", [5, 6, 7, 20])
def test_spline_reproduces_hg_disk_for_odd_and_even_knot_counts(num_knots):
    disk_params = Parameter_Index.disk_params.copy()
    disk_params.update({"sma": 60.0, "inclination": 60.0, "position_angle": 30.0})
    misc_params = Parameter_Index.misc_params.copy()

    hg_params = HenyeyGreenstein_SPF.params.copy()
    hg_params["g"] = G
    hg_img = np.array(objective_model(disk_params, hg_params, None, misc_params,
                                      ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                                      HenyeyGreenstein_SPF, None))

    spf = InterpolatedUnivariateSpline_SPF.params.copy()
    spf["num_knots"] = num_knots
    knots = np.array(InterpolatedUnivariateSpline_SPF.get_knots(spf))
    ci = InterpolatedUnivariateSpline_SPF.center_index(num_knots)
    free_knots = np.concatenate([knots[:ci], knots[ci + 1:]])
    spf["knot_values"] = jnp.array(_hg(free_knots) / _hg(0.0))

    spline_misc = misc_params.copy()
    spline_misc["flux_scaling"] = misc_params["flux_scaling"] * _hg(0.0)
    spline_img = np.array(objective_model(disk_params, spf, None, spline_misc,
                                          ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                                          InterpolatedUnivariateSpline_SPF, None))

    bright = hg_img > 1e-3 * hg_img.max()
    frac_err = np.abs(hg_img - spline_img)[bright] / hg_img[bright]
    # Coarse knot grids interpolate less accurately; the point here is that odd and
    # even counts behave alike once the centre knot is placed correctly
    tol = 0.01 if num_knots >= 20 else 0.05
    assert frac_err.max() < tol


@pytest.mark.parametrize("num_knots", [5, 6])
def test_warm_start_knots_evaluates_hg_at_the_free_knot_positions(num_knots):
    disk_params = Parameter_Index.disk_params.copy()
    spf = InterpolatedUnivariateSpline_SPF.params.copy()
    spf["num_knots"] = num_knots
    spf["knot_values"] = jnp.ones(num_knots)
    opt = Optimizer(ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                    InterpolatedUnivariateSpline_SPF, None,
                    disk_params, spf, None, Parameter_Index.misc_params.copy())
    kv = np.array(opt.warm_start_knots(init_g=G))
    knots = np.array(InterpolatedUnivariateSpline_SPF.get_knots(spf))
    ci = InterpolatedUnivariateSpline_SPF.center_index(num_knots)
    free_knots = np.concatenate([knots[:ci], knots[ci + 1:]])
    np.testing.assert_allclose(kv, _hg(free_knots) / _hg(0.0), rtol=1e-12)
