import numpy as np
import jax.numpy as jnp
from types import SimpleNamespace

from grater_jax.disk_model.objective_functions import Parameter_Index
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF
from grater_jax.optimization import optimize_framework
from grater_jax.optimization.optimize_framework import Optimizer


def _optimizer(num_knots=4):
    spf = InterpolatedUnivariateSpline_SPF.params.copy()
    spf["num_knots"] = num_knots
    spf["knot_values"] = jnp.ones(num_knots)
    return Optimizer(ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                     InterpolatedUnivariateSpline_SPF, None,
                     Parameter_Index.disk_params.copy(), spf, None, Parameter_Index.misc_params.copy())


def _capture_bounds(monkeypatch):
    seen = {}

    def fake_minimize(fun, x0, method=None, bounds=None, jac=None, options=None):
        seen["bounds"] = list(bounds)
        return SimpleNamespace(x=np.asarray(x0), success=True, fun=0.0, message="stub")

    monkeypatch.setattr(optimize_framework, "minimize", fake_minimize)
    return seen


def test_per_element_bounds_for_non_logscaled_array_param(monkeypatch):
    opt = _optimizer()
    seen = _capture_bounds(monkeypatch)
    target = np.zeros((140, 140)); err = np.ones_like(target)
    lower = [np.array([0.1, 0.2, 0.3, 0.4]), 40.0]
    upper = [np.array([1.1, 1.2, 1.3, 1.4]), 60.0]
    opt.scipy_bounded_optimize(["knot_values", "sma"], (lower, upper), [], ["knot_values"], target, err)
    assert seen["bounds"] == [(0.1, 1.1), (0.2, 1.2), (0.3, 1.3), (0.4, 1.4), (40.0, 60.0)]


def test_per_element_bounds_for_logscaled_array_param(monkeypatch):
    opt = _optimizer()
    seen = _capture_bounds(monkeypatch)
    target = np.zeros((140, 140)); err = np.ones_like(target)
    lower = [np.array([0.1, 0.2, 0.3, 0.4])]
    upper = [np.array([1.0, 2.0, 3.0, 4.0])]
    opt.scipy_bounded_optimize(["knot_values"], (lower, upper), ["knot_values"], ["knot_values"], target, err)
    got = np.array(seen["bounds"])
    np.testing.assert_allclose(got[:, 0], np.log(np.array([0.1, 0.2, 0.3, 0.4]) + 1e-14))
    np.testing.assert_allclose(got[:, 1], np.log([1.0, 2.0, 3.0, 4.0]))
