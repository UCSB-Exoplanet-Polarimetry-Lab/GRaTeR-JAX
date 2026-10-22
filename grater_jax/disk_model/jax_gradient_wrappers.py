"""
jax_gradient_wrappers.py
========================

Differentiable log-likelihood functions for disk modeling.

This module defines JAX-compiled objective functions and their gradients for
disk image generation and likelihood evaluation. It provides standard, Winnie
PSF, and spline-based variants, along with convenience wrappers for computing
per-pixel residuals and Gaussian log-likelihoods. These functions are meant
to be differentiable versions of the methods in jax_model_wrappers.py, however,
there is some overhead needed to match the parameters of the two, this is handled
by objective_grad in objective_functions.py.
"""

import numpy as np
import jax
import jax.numpy as jnp

from functools import partial
from grater_jax.disk_model.jax_model_wrappers import (
    jax_model, jax_model_spline, jax_model_winnie, jax_model_spline_winnie
)


@partial(jax.jit, static_argnames=('ll_method',))
def log_likelihood(image, target_image, err_map, ll_method='sum'):
    """
    Compute the Gaussian log-likelihood between a model image and observed data.

    Parameters
    ----------
    image : jnp.ndarray
        Model image of shape (H, W).
    target_image : jnp.ndarray
        Observed image of the same shape as `image`.
    err_map : jnp.ndarray
        Per-pixel standard deviation (noise map); same shape as `image`.
    ll_method : {'sum', 'mean'}, optional
        How to reduce the per-pixel Gaussian log-density terms across the image.
        ``'sum'`` (default) returns the total log-likelihood — the statistically
        correct choice for MCMC, since log-posterior ratios depend on absolute
        values. ``'mean'`` divides by ``jnp.size(image)`` and returns the
        per-pixel average — sometimes preferred for scipy gradient-based
        optimisation because the smaller magnitude keeps the L-BFGS-B ``ftol``
        convergence criterion from tripping prematurely on large images.

    Returns
    -------
    float
        The reduced log-likelihood assuming independent Gaussian errors.

    Raises
    ------
    ValueError
        If ``ll_method`` is not ``'sum'`` or ``'mean'``.
    """
    sigma2 = jnp.square(err_map)
    result = ((target_image - image) ** 2) / (sigma2 + 1e-40) + jnp.log(sigma2 + 1e-40)
    if ll_method == 'sum':
        reduced = jnp.sum(result)
    elif ll_method == 'mean':
        reduced = jnp.mean(result)
    else:
        raise ValueError(f"ll_method must be 'sum' or 'mean', got {ll_method!r}")
    return -0.5 * reduced


@jax.jit
def residuals(image, target_image, err_map):
    """
    Compute per-pixel squared normalized residuals plus log-variance term.

    Parameters
    ----------
    image : jnp.ndarray
        Model image of shape (H, W).
    target_image : jnp.ndarray
        Observed image of the same shape.
    err_map : jnp.ndarray
        Per-pixel standard deviation (noise map); same shape as `image`.

    Returns
    -------
    jnp.ndarray
        Residual map of the same shape as `image`.
    """
    sigma2 = jnp.square(err_map)
    result = ((target_image - image) ** 2) / (sigma2 + 1e-40) + jnp.log(sigma2 + 1e-40)
    return result

def jax_model_ll(DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params, target_image, err_map,
              throughput, flux_scaling = 1e6, distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25, ll_method = 'sum', output='image'):
    """
    Get the log likelihood for the generated disk model image given disk, scattering function, point spread function,
    stellar psf point spread function, and misceallaneous parameters along with the target image and error map. This
    function only applies when PSFModel != Winnie_PSF and FuncModel != InterpolatedUnivariateSpline_SPF. This
    function is later differentiated by jax grad.
    
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    disk_params : jnp.array
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : jnp.array
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : jnp.array
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    stellar_psf_params : jnp.array, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image.
    distance : float
        Distance to the star in pc (default 70.)
    pxInArcsec : float
        Pixel field of view in arcsec/px (default the SPHERE pixel
        scale 0.01225 arcsec/px)
    nx : int
        number of pixels along the x axis of the image (default 200)
    ny : int
        number of pixels along the y axis of the image (default 200)
    halfNbSlices : integer
        half number of distances along the line of sight
    flux_scaling : float
        Scaling factor for disk model.
            
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map.
    """

    image = jax_model(
        DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params, throughput,
        flux_scaling=flux_scaling, distance=distance, pxInArcsec=pxInArcsec,
        nx=nx, ny=ny, halfNbSlices=halfNbSlices, output=output)
    return log_likelihood(image, target_image, err_map, ll_method=ll_method)

def jax_model_winnie_ll(DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params, target_image, err_map,
                     throughput, flux_scaling = 1e6, distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25, ll_method = 'sum', output='image'):

    """
    Get the log likelihood for the generated disk model image given disk, scattering function, point spread function,
    stellar psf point spread function, and misceallaneous parameters along with the target image and error map. This
    function only applies when PSFModel == Winnie_PSF and FuncModel != InterpolatedUnivariateSpline_SPF. This
    function is later differentiated by jax grad.
    
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    winnie_psf : class (Can be found in disk_model/winnie_class.py)
        JWST off axis PSF, modeled after Winnie framework
    disk_params : jnp.array
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : jnp.array
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    stellar_psf_params : jnp.array, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image.
    distance : float
        Distance to the star in pc (default 70.)
    pxInArcsec : float
        Pixel field of view in arcsec/px (default the SPHERE pixel
        scale 0.01225 arcsec/px)
    nx : int
        number of pixels along the x axis of the image (default 200)
    ny : int
        number of pixels along the y axis of the image (default 200)
    halfNbSlices : integer
        half number of distances along the line of sight
    flux_scaling : float
        Scaling factor for disk model.
            
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map.
    """

    image = jax_model_winnie(
        DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params, throughput,
        flux_scaling=flux_scaling, distance=distance, pxInArcsec=pxInArcsec,
        nx=nx, ny=ny, halfNbSlices=halfNbSlices, output=output)
    return log_likelihood(image, target_image, err_map, ll_method=ll_method)

def jax_model_spline_ll(DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params, target_image, err_map,
                     throughput, flux_scaling = 1e6, distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25,
                     knots=np.delete(np.linspace(1,-1,5), 2), ll_method = 'sum', output='image'):

    """
    Get the log likelihood for the generated disk model image given disk, scattering function, point spread function,
    stellar psf point spread function, and misceallaneous parameters along with the target image and error map. This
    function only applies when PSFModel != Winnie_PSF and FuncModel == InterpolatedUnivariateSpline_SPF. This
    function is later differentiated by jax grad.
    
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    disk_params : jnp.array
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : jnp.array
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : jnp.array
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    stellar_psf_params : jnp.array, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image.
    distance : float
        Distance to the star in pc (default 70.)
    pxInArcsec : float
        Pixel field of view in arcsec/px (default the SPHERE pixel
        scale 0.01225 arcsec/px)
    nx : int
        number of pixels along the x axis of the image (default 200)
    ny : int
        number of pixels along the y axis of the image (default 200)
    halfNbSlices : integer
        half number of distances along the line of sight
    flux_scaling : float
        Scaling factor for disk model.
            
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map.
    """

    image = jax_model_spline(
        DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params, throughput,
        flux_scaling=flux_scaling, distance=distance, pxInArcsec=pxInArcsec,
        nx=nx, ny=ny, halfNbSlices=halfNbSlices, output=output, knots=knots)
    return log_likelihood(image, target_image, err_map, ll_method=ll_method)

def jax_model_spline_winnie_ll(DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params, target_image, err_map,
                     throughput, flux_scaling = 1e6, distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25,
                     knots=np.delete(np.linspace(1,-1,5), 2), ll_method = 'sum', output='image'):

    """
    Get the log likelihood for the generated disk model image given disk, scattering function, point spread function,
    stellar psf point spread function, and misceallaneous parameters along with the target image and error map. This
    function only applies when PSFModel == Winnie_PSF and FuncModel == InterpolatedUnivariateSpline_SPF. This
    function is later differentiated by jax grad.
    
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    winnie_psf : class (Can be found in disk_model/winnie_class.py)
        JWST off axis PSF, modeled after Winnie framework
    disk_params : jnp.array
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : jnp.array
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    stellar_psf_params : jnp.array, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image.
    distance : float
        Distance to the star in pc (default 70.)
    pxInArcsec : float
        Pixel field of view in arcsec/px (default the SPHERE pixel
        scale 0.01225 arcsec/px)
    nx : int
        number of pixels along the x axis of the image (default 200)
    ny : int
        number of pixels along the y axis of the image (default 200)
    halfNbSlices : integer
        half number of distances along the line of sight
    flux_scaling : float
        Scaling factor for disk model.
            
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map.
    """

    image = jax_model_spline_winnie(
        DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params, throughput,
        flux_scaling=flux_scaling, distance=distance, pxInArcsec=pxInArcsec,
        nx=nx, ny=ny, halfNbSlices=halfNbSlices, output=output, knots=knots)
    return log_likelihood(image, target_image, err_map, ll_method=ll_method)

# JAX GRADS (comment out whatever grad functions you don't need to save gpu memory)

"""
These methods are the differentiable versions of the jax log likelihood methods above. They return the gradients only
for the 1d parameter arrays.

Return format: (jnp.array, jnp.array, jnp.array, jnp.array)

For disk_params, spf_params, psf_params, and stellar_psf_params each flattened in the same order as their corresponding dictionaries
"""

jax_model_grad = jax.jit(jax.grad(jax_model_ll, argnums=(5, 6, 7, 8, 12)),
                                static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'PSFModel', 'StellarPSFModel',
                                                 'nx', 'ny', 'halfNbSlices', 'll_method', 'output'])

jax_model_winnie_grad = jax.jit(jax.grad(jax_model_winnie_ll, argnums=(5, 6, 7, 11)),
                                static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'winnie_psf', 'StellarPSFModel',
                                                 'nx', 'ny', 'halfNbSlices', 'll_method', 'output'])

jax_model_spline_grad = jax.jit(jax.grad(jax_model_spline_ll, argnums=(5, 6, 7, 8, 12)),
                                static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'PSFModel', 'StellarPSFModel',
                                                 'nx', 'ny', 'halfNbSlices', 'll_method', 'output'])

jax_model_spline_winnie_grad = jax.jit(jax.grad(jax_model_spline_winnie_ll, argnums=(5, 6, 7, 11)),
                                static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'winnie_psf', 'StellarPSFModel',
                                                 'nx', 'ny', 'halfNbSlices', 'll_method', 'output'])