"""Polarization geometry, PSF ordering, and inference integration."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import fftconvolve

from grater_jax.disk_model.polarization import (
    polarized_intensity_to_stokes, stokes_to_azimuthal, polarized_image,
)
from grater_jax.disk_model.SLD_ojax import ScatteredLightDisk
from grater_jax.disk_model.SLD_utils import (
    DustEllipticalDistribution2PowerLaws, HenyeyGreenstein_SPF,
    InterpolatedUnivariateSpline_SPF, EMP_PSF, GAUSSIAN_PSF,
)
from grater_jax.disk_model.objective_functions import Parameter_Index, objective_ll
from grater_jax.optimization.optimize_framework import Optimizer

jax.config.update('jax_enable_x64', True)


def make_optimizer(output='image', spline=False, psf=None):
    disk = dict(Parameter_Index.disk_params)
    disk.update(inclination=55., x_center=11.3, y_center=10.7, sma=8., ksi0=1.)
    misc = dict(Parameter_Index.misc_params)
    misc.update(nx=24, ny=22, halfNbSlices=8, flux_scaling=2., distance=20., pxInArcsec=.05)
    spf = dict(InterpolatedUnivariateSpline_SPF.params) if spline else {'g': .3}
    if spline:
        spf['knot_values'] = jnp.asarray(spf['knot_values'])
    return Optimizer(ScatteredLightDisk, DustEllipticalDistribution2PowerLaws,
                     InterpolatedUnivariateSpline_SPF if spline else HenyeyGreenstein_SPF,
                     psf, disk, spf, {} if psf is None else dict(psf.params), misc,
                     output=output)


def test_cardinal_diagonal_and_center():
    p = jnp.ones((5, 5))
    q, u = polarized_intensity_to_stokes(p, 2., 2.)
    assert q[2, 3] == -1  # radial +x, polarization along y
    assert q[3, 2] == 1
    assert u[3, 3] == -1
    assert u[1, 3] == 1
    assert q[2, 2] == u[2, 2] == 0
    qp, up = stokes_to_azimuthal(q, u, 2., 2.)
    expected = np.ones((5, 5)); expected[2, 2] = 0
    np.testing.assert_allclose(qp, expected)
    np.testing.assert_allclose(up, 0, atol=1e-15)
    grad = jax.grad(lambda x: jnp.sum(polarized_intensity_to_stokes(p, x, 2.)[0]))(2.)
    assert np.isfinite(grad)


def test_convolution_before_azimuthal_transform():
    p = jnp.zeros((15, 15)).at[7, 8].set(1.).at[8, 7].set(1.)
    kernel = np.ones((5, 5)) / 25
    class Kernel:
        @staticmethod
        def generate(image, params):
            return jax.scipy.signal.fftconvolve(image, jnp.array(kernel), mode='same')
    y, x = np.indices(p.shape)
    phi = np.arctan2(y - 7., x - 7.)
    q = -np.asarray(p) * np.cos(2 * phi)
    u = -np.asarray(p) * np.sin(2 * phi)
    qc, uc = fftconvolve(q, kernel, mode='same'), fftconvolve(u, kernel, mode='same')
    qp = -qc * np.cos(2 * phi) - uc * np.sin(2 * phi)
    up = qc * np.sin(2 * phi) - uc * np.cos(2 * phi)
    qp[7, 7] = up[7, 7] = 0
    for name, expected in [('Q', qc), ('U', uc), ('Qphi', qp), ('Uphi', up)]:
        actual = polarized_image(p, 7., 7., Kernel, None, 1., 1., None, name)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
    assert np.max(np.abs(up)) > .01
    assert not np.allclose(qp, fftconvolve(p, kernel, mode='same'))


@pytest.mark.parametrize('spline', [False, True])
def test_intrinsic_model_products(spline):
    opt = make_optimizer(spline=spline)
    p = opt.get_model()
    q, u = opt.get_model('Q'), opt.get_model('U')
    np.testing.assert_allclose(q*q + u*u, p*p, atol=1e-13)
    np.testing.assert_allclose(opt.get_model('Qphi'), p, atol=1e-13)
    np.testing.assert_allclose(opt.get_model('Uphi'), 0, atol=1e-13)
    np.testing.assert_allclose(opt.get_model('PI'), p, atol=1e-13)
    np.testing.assert_array_equal(opt.get_model(), p)


@pytest.mark.parametrize('spline', [False, True])
@pytest.mark.parametrize('output', ['Q', 'U', 'Qphi', 'Uphi', 'PI'])
def test_likelihood_and_gradient(spline, output):
    opt = make_optimizer(output, spline, GAUSSIAN_PSF)
    target = opt.get_model() * 1.2
    err = jnp.ones_like(target) * .01
    keys = ['g'] if not spline else ['flux_scaling']
    initial = opt.spf_params['g'] if not spline else opt.misc_params['flux_scaling']
    def fun(value):
        return float(opt.get_objective_likelihood([value], keys, target, err))
    eps = 1e-5
    numeric = (fun(initial + eps) - fun(initial - eps)) / (2*eps)
    analytic = opt.get_objective_gradient([initial], keys, target, err)[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-8)
    ll = objective_ll(opt.disk_params, opt.spf_params, opt.psf_params, None,
                      opt.misc_params, opt.DiskModel, opt.DistrModel, opt.FuncModel,
                      opt.PSFModel, None, target, err, output=output)
    np.testing.assert_allclose(ll, fun(initial))


def test_empirical_psf_and_throughput():
    opt = make_optimizer('Q', psf=EMP_PSF)
    kernel = np.ones((3, 3)) / 9
    intrinsic = make_optimizer().get_model('Q')
    opt.set_empirical_psf(kernel)
    expected = fftconvolve(intrinsic, kernel, mode='same')
    opt.set_throughput(np.full((22, 24), .7))
    np.testing.assert_allclose(opt.get_model(), expected * .7, atol=1e-13)


def test_invalid_configuration_and_gaussian_offset():
    opt = make_optimizer(psf=GAUSSIAN_PSF)
    baseline = opt.get_model('Q')
    opt.psf_params['offset'] = 25.
    np.testing.assert_allclose(opt.get_model('Q'), baseline, atol=1e-13)
    with pytest.raises(ValueError, match='output must'):
        opt.get_model('invalid')
    opt.StellarPSFModel = object
    opt.stellar_psf_params = 0.
    with pytest.raises(ValueError, match='StellarPSFModel=None'):
        opt.get_model('Q')


def test_center_gradient_and_fit_recovery():
    opt = make_optimizer('Qphi', psf=GAUSSIAN_PSF)
    target = opt.get_model() * 1.5
    err = jnp.ones_like(target) * .001
    initial = opt.disk_params['x_center']
    eps = 1e-5
    numeric = (opt.get_objective_likelihood([initial + eps], ['x_center'], target, err)
               - opt.get_objective_likelihood([initial - eps], ['x_center'], target, err)) / (2*eps)
    analytic = opt.get_objective_gradient([initial], ['x_center'], target, err)[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-7)
    result = opt.scipy_bounded_optimize(
        ['flux_scaling'], ([.5], [5.]), [], [], target, err,
        use_grad=True, iters=50, ftol=1e-12, gtol=1e-10)
    assert result.success
    np.testing.assert_allclose(opt.misc_params['flux_scaling'], 3., rtol=1e-5)


def test_gradient_uses_throughput():
    opt = make_optimizer('Q')
    opt.set_throughput(np.full((22, 24), .4))
    target = opt.get_model() * 1.1
    err = jnp.ones_like(target) * .01
    numeric = (opt.get_objective_likelihood([.30001], ['g'], target, err)
               - opt.get_objective_likelihood([.29999], ['g'], target, err)) / .00002
    analytic = opt.get_gradient(['g'], target, err)[0]
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-9)


@pytest.mark.parametrize('spline', [False, True], ids=['hg', 'spline'])
def test_disk_polarization_matches_numpy_reference(spline, tmp_path):
    """Check the public API against independent NumPy/SciPy calculations.

    Like the existing disk comparison tests, save a reference/model figure.
    pytest's tmp_path keeps generated plots out of tracked test_results files.
    """
    import matplotlib.pyplot as plt

    opt = make_optimizer(spline=spline)
    polarized_intensity = np.asarray(opt.get_model())
    ny, nx = polarized_intensity.shape
    y, x = np.indices((ny, nx))
    phi = np.arctan2(y - opt.disk_params['y_center'],
                     x - opt.disk_params['x_center'])
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    q_intrinsic = -polarized_intensity * c
    u_intrinsic = -polarized_intensity * s

    ky, kx = np.indices((5, 5))
    kernel = np.exp(-((kx - 2)**2 + (ky - 2)**2) / 2.)
    kernel /= kernel.sum()
    # A distinct class keeps this kernel isolated from EMP_PSF's global cache.
    class ReferencePSF(EMP_PSF):
        img = jnp.asarray(kernel)

    opt.PSFModel = ReferencePSF
    opt.psf_params = dict(ReferencePSF.params)
    throughput = .6 + .3 * x / (nx - 1)
    opt.throughput = jnp.asarray(throughput)
    q = fftconvolve(q_intrinsic, kernel, mode='same') * throughput
    u = fftconvolve(u_intrinsic, kernel, mode='same') * throughput
    expected = {
        'Q': q,
        'U': u,
        'Qphi': -q * c - u * s,
        'Uphi': q * s - u * c,
        'PI': np.hypot(q, u),
    }
    actual = {name: np.asarray(opt.get_model(output=name)) for name in expected}
    for name in expected:
        assert actual[name].shape == (ny, nx)
        assert np.isfinite(actual[name]).all()
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-13,
                                   err_msg=f'{name} differs from the reference')

    # This disk must distinguish Stokes convolution from scalar-image blurring.
    blurred_p = fftconvolve(polarized_intensity, kernel, mode='same') * throughput
    assert np.max(np.abs(expected['Qphi'] - blurred_p)) > 1e-4 * blurred_p.max()
    assert np.max(np.abs(expected['Uphi'])) > 1e-4 * blurred_p.max()

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    try:
        for col, name in enumerate(expected):
            limit = np.max(np.abs(expected[name]))
            for row, images in enumerate((expected, actual)):
                axes[row, col].imshow(images[name], origin='lower', cmap='RdBu_r',
                                      vmin=-limit, vmax=limit)
                axes[row, col].set_title(f'{"Reference" if row == 0 else "GRaTeR-JAX"} {name}')
        fig.savefig(tmp_path / 'polarization_reference_comparison.png', dpi=150)
    finally:
        plt.close(fig)
