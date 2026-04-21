"""Unit tests for MCMC_model helpers."""
import numpy as np
import pytest

from grater_jax.optimization.mcmc_model import _robust_corner_ranges


def test_robust_corner_ranges_crops_outliers():
    """A few straggling walkers do not stretch the returned window."""
    rng = np.random.default_rng(0)
    core = rng.normal(loc=0.0, scale=1.0, size=(10_000, 1))
    outliers = np.array([[1e6], [-1e6], [5e5]])
    samples = np.concatenate([core, outliers], axis=0)

    ranges = _robust_corner_ranges(samples)
    assert len(ranges) == 1
    lo, hi = ranges[0]
    assert abs(lo) < 10.0
    assert abs(hi) < 10.0
    assert lo < -1.0 < 1.0 < hi  # core ±1σ still inside window


def test_robust_corner_ranges_matches_percentiles_with_padding():
    """Window equals the requested percentiles ± pad * span."""
    samples = np.linspace(0.0, 100.0, 10_001).reshape(-1, 1)
    ranges = _robust_corner_ranges(samples, lo=10.0, hi=90.0, pad=0.1)
    lo, hi = ranges[0]
    # 10th pct = 10, 90th pct = 90; span = 80; pad = 8
    assert lo == pytest.approx(10.0 - 8.0, abs=0.1)
    assert hi == pytest.approx(90.0 + 8.0, abs=0.1)


def test_robust_corner_ranges_handles_degenerate_column():
    """A constant column returns a finite non-empty window around the median."""
    samples = np.column_stack([
        np.random.default_rng(1).normal(size=1000),
        np.full(1000, 3.5),  # degenerate
        np.zeros(1000),       # degenerate at zero
    ])
    ranges = _robust_corner_ranges(samples)
    assert len(ranges) == 3
    for lo, hi in ranges:
        assert np.isfinite(lo) and np.isfinite(hi)
        assert hi > lo

    # Degenerate columns get a tiny symmetric window around the median
    lo1, hi1 = ranges[1]
    assert lo1 < 3.5 < hi1
    lo2, hi2 = ranges[2]
    assert lo2 < 0.0 < hi2


def test_robust_corner_ranges_output_shape():
    """Output length matches the number of columns."""
    samples = np.random.default_rng(2).normal(size=(500, 7))
    ranges = _robust_corner_ranges(samples)
    assert len(ranges) == 7
    for bounds in ranges:
        assert len(bounds) == 2
