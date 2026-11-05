import numpy as np
import jax

from grater_jax.disk_model.objective_functions import objective_model, Parameter_Index
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF

jax.config.update("jax_enable_x64", True)


def _image(disk_params):
    spf_params = HenyeyGreenstein_SPF.params.copy()
    misc_params = Parameter_Index.misc_params.copy()
    return np.array(objective_model(
        disk_params, spf_params, None, misc_params,
        ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF, None,
    ))


def _eccentric_disk(omega):
    d = Parameter_Index.disk_params.copy()
    d.update({"sma": 40.0, "e": 0.3, "inclination": 30.0, "position_angle": 20.0, "omega": omega})
    return d


def test_omega_is_read_from_the_template_position():
    # Rotating the pericenter of an eccentric disk must change the image
    img0 = _image(_eccentric_disk(0.0))
    img90 = _image(_eccentric_disk(90.0))
    assert np.isfinite(img0).all() and np.isfinite(img90).all()
    assert np.max(np.abs(img0 - img90)) > 1e-3 * img0.max()


def test_extra_disk_param_keys_do_not_change_the_image():
    # Callers (e.g. the SPF tutorial) append keys like halfNbSlices to disk_params;
    # positional reads must not pick those up as omega
    base = _eccentric_disk(90.0)
    extra = dict(base)
    extra["halfNbSlices"] = 25
    np.testing.assert_allclose(_image(extra), _image(base), rtol=0, atol=0)
