import numpy as np
import pytest

from grater_jax.optimization.optimize_framework import OptimizeUtils


def _ramp_image(n=300):
    # Distinct values everywhere so a wrong crop is detectable
    return np.arange(n * n, dtype=float).reshape(n, n)


def test_process_image_square_crop_matches_slice():
    img = _ramp_image()
    out = OptimizeUtils.process_image(img, bounds=(70, 210, 70, 210))
    assert out.shape == (140, 140)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, img[70:210, 70:210].astype(np.float32))


def test_process_image_replaces_nans_with_zero():
    img = _ramp_image()
    img[100, 100] = np.nan
    out = OptimizeUtils.process_image(img, bounds=(70, 210, 70, 210))
    assert np.isfinite(out).all()
    assert out[30, 30] == 0.0


def test_process_image_rejects_non_square_bounds():
    img = _ramp_image()
    with pytest.raises(NotImplementedError):
        OptimizeUtils.process_image(img, bounds=(70, 210, 80, 220))
    with pytest.raises(NotImplementedError):
        OptimizeUtils.process_image(img, bounds=(70, 210, 70, 200))


def test_process_image_rejects_scale_factor():
    img = _ramp_image()
    with pytest.raises(NotImplementedError):
        OptimizeUtils.process_image(img, scale_factor=2, bounds=(70, 210, 70, 210))
