"""
objective_functions.py
======================

Objective functions for disk modeling and optimization.

This module provides functions that connect high-level optimization routines
to the underlying JAX-accelerated disk model wrappers. It defines parameter
templates, packing utilities, and objective functions for model evaluation,
log-likelihood computation, and gradient calculation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from grater_jax.disk_model.SLD_utils import InterpolatedUnivariateSpline_SPF, Winnie_PSF
from grater_jax.disk_model.jax_model_wrappers import jax_model, jax_model_spline, jax_model_winnie, jax_model_spline_winnie
from grater_jax.disk_model.jax_gradient_wrappers import jax_model_grad, jax_model_spline_grad, jax_model_winnie_grad, jax_model_spline_winnie_grad, log_likelihood
import matplotlib.pyplot as plt

class Parameter_Index:
    """
    Default parameter sets for disk modeling components.

    This class defines canonical parameter dictionaries used for optimization
    and simulation of scattered light disks. These parameters act as templates
    for packing and fitting routines and represent all modifiable physical and
    observational quantities.

    Attributes
    ----------
    disk_params : dict
        Dictionary of physical parameters describing the disk geometry and density:
            - accuracy : float
                Numerical accuracy for integrators.
            - alpha_in, alpha_out : float
                Inner and outer radial density power-law slopes.
            - sma : float
                Semi-major axis (in pixels or AU, depending on model).
            - e : float
                Eccentricity of the disk.
            - ksi0 : float
                Azimuthal anisotropy parameter.
            - gamma, beta : float
                Vertical structure parameters.
            - rmin : float
                Inner radius cutoff.
            - dens_at_r0 : float
                Density normalization at reference radius.
            - inclination : float
                Inclination angle (degrees).
            - position_angle : float
                Disk orientation on sky (degrees).
            - x_center, y_center : float
                Image center coordinates.
            - halfNbSlices : int
                Number of angular slices for 3D integration (half of total).
            - omega : float
                Argument of pericenter (in radians or degrees).

    misc_params : dict
        Dictionary of observational and image grid parameters:
            - distance : float
                Distance to the system (in parsecs).
            - pxInArcsec : float
                Pixel scale (arcseconds per pixel).
            - nx, ny : int
                Image dimensions (width and height in pixels).
            - halfNbSlices : int
                Number of angular slices for rendering (should match disk_params).
            - flux_scaling : float
                Normalization factor for model brightness.

    Notes
    -----
    Parameter dictionaries for scattering phase functions (SPFs), PSFs, and stellar PSFs
    are defined in `SLD_utils.py`. Spline SPFs and Winnie PSFs use instantiated
    model classes (e.g., `InterpolatedUnivariateSpline_SPF`, `Winnie_PSF`).
    """
    
    disk_params = {'accuracy': 5.e-3, 'alpha_in': 5, 'alpha_out': -5, 'sma': 50, 'e': 0., 'ksi0': 3., 'gamma': 2., 'beta': 1., 'rmin': 0.,
                'dens_at_r0': 1., 'inclination': 0, 'position_angle': 0, 'x_center': 70., 'y_center': 70., 'halfNbSlices': 25, 'omega': 0.,}

    misc_params = {'distance': 50., 'pxInArcsec': 0.01414, 'nx': 140, 'ny': 140, 'halfNbSlices': 25, 'flux_scaling': 1e6}


def pack_pars(p_dict, orig_dict):
    """
    Pack parameter values from a dictionary into a JAX array.

    The output array follows the key order defined in `orig_dict`, which is
    typically a template dictionary that defines the parameter structure.
    This is how jax classes are wrapped in the JAX code.

    Parameters
    ----------
    p_dict : dict
        Dictionary of parameter values to be packed. Keys must match those in `orig_dict`.
    orig_dict : dict
        Reference dictionary that defines the desired key ordering.

    Returns
    -------
    jnp.ndarray
        JAX array of parameter values in the order defined by `orig_dict`.
    """
    p_arrs = []
    for name in orig_dict.keys():
        p_arrs.append(p_dict[name])
    return jnp.asarray(p_arrs)


"""
These objective functions serve as the middleware that connects the Optimizer class to the lower level JAX code.
"""

def objective_model(disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, stellar_psf_params=None, StellarPSFModel=None,
                       throughput=None, **kwargs):

    """
    Generate a disk model image given disk, scattering function, point spread function, stellar psf point spread function,
    and misceallaneous parameters.

    disk_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    misc_params : dict of (str, float) pairs
        The parameter dictionary for misceallanious values, such as image size and flux scaling, dictionary is
        made of (parameter name, parameter value) pairs.
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    throughput : np.ndarray, optional
        The throughput image to apply to the generated disk image. Defaults to None, indicating all 1s.
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
        
    Returns
    -------
    jnp.ndarray
        Generated disk model image
    """

    if throughput is None:
        throughput = jnp.ones((misc_params['ny'], misc_params['nx']))

    if StellarPSFModel is None:
        stellar_psf_params = 0.
    if PSFModel is None:
        psf_params = 0.
    
    if not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel != Winnie_PSF:
        model_image = jax_model(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(spf_params) if isinstance(spf_params, dict) else spf_params,
            PSFModel.pack_pars(psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel == Winnie_PSF:
        model_image = jax_model_winnie(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(spf_params) if isinstance(spf_params, dict) else spf_params,
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif issubclass(FuncModel, InterpolatedUnivariateSpline_SPF) and PSFModel != Winnie_PSF:
        model_image = jax_model_spline(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            spf_params['knot_values'],
            PSFModel.pack_pars(psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(spf_params)
        )
    else:
        model_image = jax_model_spline_winnie(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            spf_params['knot_values'],
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(spf_params)
        )

    return model_image

def objective_ll(disk_params, spf_params, psf_params, stellar_psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, target_image, err_map,
                       throughput = None, **kwargs):
    """
    Get the log likelihood for the generated disk model image given disk, scattering function, point spread function,
    stellar psf point spread function, and misceallaneous parameters along with the target image and error map.

    disk_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    misc_params : dict of (str, float) pairs
        The parameter dictionary for misceallanious values, such as image size and flux scaling, dictionary is
        made of (parameter name, parameter value) pairs.
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image. Defaults to None, indicating all 1s.
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
        
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map
    """

    model_image = objective_model(
        disk_params, spf_params, psf_params, stellar_psf_params, misc_params, DiskModel, DistrModel, FuncModel, PSFModel,
        StellarPSFModel, throughput=throughput, **kwargs
    )

    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)

    return -0.5 * jnp.sum(result)  # / jnp.size(target_image)

def objective_fit(params_fit, fit_keys, disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, target_image, err_map,
                       throughput = None, stellar_psf_params = None, StellarPSFModel = None, **kwargs):
    """
    Same as the objective_ll function but accepts replacement values for the given parameters in fit_keys.
    This is ideal for fitting as it provides a clean objective function for a given set of parameters.

    ex: fit_keys = ['alpha_in', 'alpha_out'], params_fit = [-5., 5.], the function will compute the
    log-likelihood for the given parameters while replacing alpha_in with -5 and alpha_out with 5.

    params_fit : list of float
        The replacement values for the parameters indicated by fit_keys.
    fit_keys : list of str
        The names of the parameters to have their values be replaced by the corresponding values in params_fit.
    disk_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    misc_params : dict of (str, float) pairs
        The parameter dictionary for misceallanious values, such as image size and flux scaling, dictionary is
        made of (parameter name, parameter value) pairs.
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image. Defaults to None, indicating all 1s.
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
        
    Returns
    -------
    float
        Log likelihood for the generated disk image, target image, and error map
    """

    if throughput is None:
        throughput = jnp.ones((misc_params['ny'], misc_params['nx']))

    if StellarPSFModel is None:
        stellar_psf_params = 0.
    if PSFModel is None:
        psf_params = 0.

    # These temporary dictionaries are edited based on params_fit
    temp_disk_params = disk_params.copy() if isinstance(disk_params, dict) else {0.}
    temp_spf_params = spf_params.copy() if isinstance(spf_params, dict) else {0.}
    temp_psf_params = psf_params.copy() if isinstance(psf_params, dict) else {0.}
    temp_stellar_psf_params = stellar_psf_params.copy() if isinstance(stellar_psf_params, dict) else {0.}
    temp_misc_params = misc_params.copy() if isinstance(misc_params, dict) else {0.}

    # Corresponding index of params_fit for each key in fit_keys
    param_index = 0

    for key in fit_keys:
        if key in temp_disk_params:
            temp_disk_params[key] = params_fit[param_index]
        elif key in temp_spf_params:
            temp_spf_params[key] = params_fit[param_index]
        elif key in temp_psf_params:
            temp_psf_params[key] = params_fit[param_index]
        elif key in temp_stellar_psf_params:
            temp_stellar_psf_params[key] = params_fit[param_index]
        elif key in temp_misc_params:
            temp_misc_params[key] = params_fit[param_index]
        param_index += 1

    if issubclass(FuncModel, InterpolatedUnivariateSpline_SPF):
        phase_fn = InterpolatedUnivariateSpline_SPF.pack_pars(
            temp_spf_params["knot_values"], knots=InterpolatedUnivariateSpline_SPF.get_knots(temp_spf_params)
        )
        if jnp.any(phase_fn(jnp.linspace(-1, 1, 200)) < 0):
            return -1e10

    if not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel != Winnie_PSF:
        model_image = jax_model(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else spf_params,
            PSFModel.pack_pars(temp_psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel == Winnie_PSF:
        model_image = jax_model_winnie(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else temp_spf_params['knot_values'],
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif issubclass(FuncModel, InterpolatedUnivariateSpline_SPF) and PSFModel != Winnie_PSF:
        model_image = jax_model_spline(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            temp_spf_params['knot_values'],
            PSFModel.pack_pars(temp_psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(temp_spf_params)
        )
    else:
        model_image = jax_model_spline_winnie(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            temp_spf_params['knot_values'],
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            throughput, distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(temp_spf_params)
        )

    return log_likelihood(model_image, target_image, err_map)


def objective_grad(disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, target_image, err_map,
                       throughput = None, stellar_psf_params = None, StellarPSFModel = None,
                        **kwargs):
    """
    Get the gradient of each parameter with respect to the log likelihood for the generated disk model image given
    disk, scattering function, point spread function, stellar psf point spread function, and misceallaneous parameters
    along with the target image and error map.

    disk_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    misc_params : dict of (str, float) pairs
        The parameter dictionary for misceallanious values, such as image size and flux scaling, dictionary is
        made of (parameter name, parameter value) pairs.
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image. Defaults to None, indicating all
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
        
    Returns
    -------
    list of jnp array
        Gradients of all the parameters with the format (gradients of disk params, gradients of spf params,
        gradients of psf_params, gradients of stellar psf params)

        Note: if the psf model is a WinniePSF, the gradients of psf_params will not be included in the output.
        Note: not all parameters are supported for gradient evaluation due to limitations of the JAX model.
        Note: the raw gradient output is transformed in the Optimizer class which wraps this method nicely.
    """

    if throughput is None:
        throughput = jnp.ones((misc_params['ny'], misc_params['nx']))
    
    if StellarPSFModel is None:
        stellar_psf_params = 0.
    if PSFModel is None:
        psf_params = 0.

    if not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel != Winnie_PSF:
        gradients = jax_model_grad(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(spf_params) if isinstance(spf_params, dict) else spf_params,
            PSFModel.pack_pars(psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel == Winnie_PSF:
        gradients = jax_model_winnie_grad(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(spf_params) if isinstance(spf_params, dict) else spf_params['knot_values'],
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif issubclass(FuncModel, InterpolatedUnivariateSpline_SPF) and PSFModel != Winnie_PSF:

        gradients = jax_model_spline_grad(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            spf_params['knot_values'],
            PSFModel.pack_pars(psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(spf_params)
        )
    else:
        gradients = jax_model_spline_winnie_grad(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            spf_params['knot_values'],
            StellarPSFModel.pack_pars(stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(spf_params)
        )

    return gradients

def objective_fit_grad(params_fit, fit_keys, disk_params, spf_params, psf_params, misc_params,
                       DiskModel, DistrModel, FuncModel, PSFModel, target_image, err_map, throughput = None,
                       stellar_psf_params = None, StellarPSFModel = None, **kwargs):
    """
    Get the gradient of each parameter with respect to the log likelihood for the generated disk model image given
    disk, scattering function, point spread function, stellar psf point spread function, and misceallaneous parameters
    along with the target image and error map.

    ex: fit_keys = ['alpha_in', 'alpha_out'], params_fit = [-5., 5.], the function will compute the
    gradients for all the parameters with respect to the log-likelihood for the given parameters
    while replacing alpha_in with -5 and alpha_out with 5.

    params_fit : list of float
        The replacement values for the parameters indicated by fit_keys.
    fit_keys : list of str
        The names of the parameters to have their values be replaced by the corresponding values in params_fit.
    disk_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the disk model, dictionary is made of
        (parameter name, parameter value) pairs.
    spf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the scattering phase function, dictionary is made of
        (parameter name, parameter value) pairs.
    psf_params : dict of (str, float) pairs
        The corresponding parameter dictionary for the point spread function, dictionary is made of
        (parameter name, parameter value) pairs.
    misc_params : dict of (str, float) pairs
        The parameter dictionary for misceallanious values, such as image size and flux scaling, dictionary is
        made of (parameter name, parameter value) pairs.
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    target_image : np.ndarray
        The target image that the log likelihood is being computed for.
    err_map : np.ndarray
        The error map for the target image.
    throughput : np.ndarray
        The throughput image to apply to the generated disk image. Defaults to None, indicating all
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
        
    Returns
    -------
    list of jnp array
        Gradients of all the parameters with the format (gradients of disk params, gradients of spf params,
        gradients of psf_params, gradients of stellar psf params)

        Note: if the psf model is a WinniePSF, the gradients of psf_params will not be included in the output.
        Note: not all parameters are supported for gradient  evaluation due to limitations of the JAX model.
        Note: the raw gradient output is transformed in the Optimizer class which wraps this method nicely.
    """

    if throughput is None:
        throughput = jnp.ones((misc_params['ny'], misc_params['nx']))

    if StellarPSFModel is None:
        stellar_psf_params = 0.
    if PSFModel is None:
        psf_params = 0.

    # These temporary dictionaries are edited based on params_fit
    temp_disk_params = disk_params.copy() if isinstance(disk_params, dict) else {0.}
    temp_spf_params = spf_params.copy() if isinstance(spf_params, dict) else {0.}
    temp_psf_params = psf_params.copy() if isinstance(psf_params, dict) else {0.}
    temp_stellar_psf_params = stellar_psf_params.copy() if isinstance(stellar_psf_params, dict) else {0.}
    temp_misc_params = misc_params.copy() if isinstance(misc_params, dict) else {0.}

    # Corresponding index of params_fit for each key in fit_keys
    param_index = 0

    for key in fit_keys:
        if key in temp_disk_params:
            temp_disk_params[key] = params_fit[param_index]
        elif key in temp_spf_params:
            temp_spf_params[key] = params_fit[param_index]
        elif key in temp_psf_params:
            temp_psf_params[key] = params_fit[param_index]
        elif key in temp_stellar_psf_params:
            temp_stellar_psf_params[key] = params_fit[param_index]
        elif key in temp_misc_params:
            temp_misc_params[key] = params_fit[param_index]
        param_index += 1

    if not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel != Winnie_PSF:
        gradients = jax_model_grad(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else spf_params,
            PSFModel.pack_pars(temp_psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif not(issubclass(FuncModel, InterpolatedUnivariateSpline_SPF)) and PSFModel == Winnie_PSF:
        gradients = jax_model_winnie_grad(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            FuncModel.pack_pars(temp_spf_params) if isinstance(spf_params, dict) else temp_spf_params['knot_values'],
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling']
        )
    elif issubclass(FuncModel, InterpolatedUnivariateSpline_SPF) and PSFModel != Winnie_PSF:

        gradients = jax_model_spline_grad(
            DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            temp_spf_params['knot_values'],
            PSFModel.pack_pars(temp_psf_params) if isinstance(psf_params, dict) else psf_params,
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(temp_spf_params)
        )
    else:
        gradients = jax_model_spline_winnie_grad(
            DiskModel, DistrModel, FuncModel, psf_params, StellarPSFModel,
            pack_pars(temp_disk_params, disk_params) if isinstance(disk_params, dict) else disk_params,
            temp_spf_params['knot_values'],
            StellarPSFModel.pack_pars(temp_stellar_psf_params) if isinstance(stellar_psf_params, dict) else stellar_psf_params,
            target_image, err_map, throughput,
            distance = misc_params['distance'], pxInArcsec = misc_params['pxInArcsec'],
            nx = misc_params['nx'], ny = misc_params['ny'], halfNbSlices=misc_params['halfNbSlices'],
            flux_scaling=misc_params['flux_scaling'], knots=FuncModel.get_knots(temp_spf_params)
        )

    return gradients