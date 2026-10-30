import pytest

from grater_jax.disk_model.objective_functions import Parameter_Index
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF, DoubleHenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF, GAUSSIAN_PSF, EMP_PSF, LinearStellarPSF, PositionalStellarPSF,
)

DEFAULTS = [
    Parameter_Index.disk_params, Parameter_Index.misc_params,
    DustEllipticalDistribution2PowerLaws.params, HenyeyGreenstein_SPF.params,
    DoubleHenyeyGreenstein_SPF.params, InterpolatedUnivariateSpline_SPF.params,
    GAUSSIAN_PSF.params, EMP_PSF.params, LinearStellarPSF.params, PositionalStellarPSF.params,
]


@pytest.mark.parametrize("defaults", DEFAULTS)
def test_default_params_cannot_be_mutated(defaults):
    key = next(iter(defaults))
    with pytest.raises(TypeError):
        defaults[key] = 12345.0
    with pytest.raises(TypeError):
        defaults["brand_new_key"] = 1.0


@pytest.mark.parametrize("defaults", DEFAULTS)
def test_default_params_copy_is_a_plain_editable_dict(defaults):
    d = defaults.copy()
    assert type(d) is dict
    d["x"] = 1.0
    assert "x" not in defaults
