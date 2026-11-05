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


def test_get_mask_matches_error_map_exclusions():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(141, 141))
    outliers = [(10, 20), (130, 5)]
    err = OptimizeUtils.create_empirical_err_map(data, mask_rad=9, outlier_pixels=outliers)
    mask = OptimizeUtils.get_mask(data, mask_rad=9, outlier_pixels=outliers)
    # The error map inflates the central disk to 1e10 and outliers by 1e6
    expected = err >= 1e5
    np.testing.assert_array_equal(mask, expected)
    assert mask.dtype == bool


def test_get_mask_central_radius():
    data = np.zeros((141, 141))
    mask = OptimizeUtils.get_mask(data, mask_rad=9)
    assert mask[70, 70]
    assert mask[70, 79] and not mask[70, 80]
    assert mask.sum() == np.sum(OptimizeUtils._radial_distance(data.shape) <= 9)
