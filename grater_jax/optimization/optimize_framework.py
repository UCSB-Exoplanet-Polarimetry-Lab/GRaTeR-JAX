"""
optimizer.py
============

High-level optimization interface for GRaTeR-JAX.

This module defines the `Optimizer` class, a top-level wrapper for the GRaTeR-JAX
framework that integrates disk morphology, scattering phase functions, PSFs, and
stellar PSFs into a single object. It provides unified methods for generating
disk models, computing likelihoods and gradients, and performing optimization
with both deterministic (SciPy) and stochastic (MCMC) approaches.

Main features
-------------
- `Optimizer.get_model` : Build a model image from current parameters.
- `Optimizer.get_objective_likelihood`, `get_objective_gradient` :
  Evaluate log-likelihood and its gradient.
- `Optimizer.scipy_optimize`, `Optimizer.scipy_bounded_optimize` :
  Run SciPy-based minimization with optional analytic gradients.
- `Optimizer.mcmc` : Run emcee-based MCMC sampling for posterior inference.
- `Optimizer.initialize_knots`, `scale_spline_to_fixed_point` :
  Utilities for initializing and normalizing spline-based SPF models.
- `Optimizer.save_human_readable`, `save_machine_readable`, `load_machine_readable` :
  Save/load model parameters for reproducibility.
- `OptimizeUtils` :
  Helper utilities for empirical error maps, image preprocessing, and
  post-processing MCMC chains.
"""

import jax
import jax.numpy as jnp
from grater_jax.disk_model.SLD_utils import *
from scipy.optimize import minimize, check_grad
from grater_jax.optimization.mcmc_model import MCMC_model
from grater_jax.disk_model.objective_functions import objective_model, objective_ll, objective_fit, log_likelihood, objective_grad, objective_fit_grad
import json
import numpy as np
from grater_jax.disk_model.SLD_utils import *

# Built for new objective function
class Optimizer:
    """
    The Optimizer class is the high level wrapper for the GRaTeR-JAX framework and all of its functionality.
    It handles all of the supported parameters for the disk morphology, scattering phase function, off axis
    point spread function, miscellaneous, and on axis stelllar psf parameters. The optimizer class is meant
    to update in place as the user uses various optimization and modification methods. To save a current
    disk model before using an optimization method, you can save a copy of the current Optimizer object.

    Parameters
    ----------
    DiskModel : class (ScatteredLighDisk is the only supported disk model)
        The disk model type
    DistrModel : class (DustEllipticalDistribution2PowerLaws is the only supported dust distribution model)
        The dust distribution model type
    FuncModel : class (Can be found in disk_model/SLD_utils.py)
        The scattering phase function model type
    PSFModel : class (Can be found in disk_model/SLD_utils.py)
        The point spread function model type
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
    StellarPSFModel : class, optional
        The scattering phase function model type, set to None be default indicating no stellar psf model.
    stellar_psf_params : dict of (str, float) pairs, optional
        The corresponding parameter dictionary for the on axis stellar psf model, dictionary is made of
        (parameter name, parameter value) pairs.
    empirical_psf_image : np.ndarray
        Image that the model uses for the empirical psf.
    kwargs : dict, optional
        Additional keyword arguments that are passed into the objective model function.
    """
    def __init__(self, DiskModel, DistrModel, FuncModel, PSFModel, disk_params, spf_params, psf_params,
                 misc_params, StellarPSFModel = None, stellar_psf_params = None, empirical_psf_image = None,
                 throughput = None,
                 **kwargs):
        self.DiskModel = DiskModel
        self.DistrModel = DistrModel
        self.FuncModel = FuncModel
        self.PSFModel = PSFModel
        self.StellarPSFModel = StellarPSFModel
        self.disk_params = disk_params
        self.spf_params = spf_params
        self.psf_params = psf_params
        self.stellar_psf_params = stellar_psf_params
        self.misc_params = misc_params
        self.kwargs = kwargs
        self.name = 'test'
        self.last_fit = None
        self.throughput = throughput

        EMP_PSF.img = empirical_psf_image

    def jit_compile_model(self):
        """
        Just-in-time compiles the disk model. Need to call again if any of the component classes are changed.
        """
        self.get_model()

    def jit_compile_gradient(self, target_image, err_map):
        """
        Just-in-time compiles the gradient of the disk model for a given target image and err_map.
        """
        self.get_gradient([], target_image, err_map)

    def set_empirical_psf(self, empirical_psf_image):
        """
        Sets EMP_PSF image to given image.
        """
        EMP_PSF.img = empirical_psf_image

    def set_throughput(self, throughput):
        """
        Sets the throughput image to the given image.

        Parameters
        ----------
        throughput : numpy.ndarray
            2D array representing the throughput image.
        Raises
        ------
        ValueError
            If the shape of the throughput image does not match the specified image dimensions in misc_params.
        """
        if jnp.shape(throughput) != (self.misc_params['ny'], self.misc_params['nx']):
            raise ValueError("Throughput image shape does not match the specified image dimensions in misc_params.")
        self.throughput = throughput

    def get_model(self):
        """
        Returns the disk model as per the current parameters.

        Returns
        -------
        numpy.ndarray
            2d image of the generated disk model
        """
        return objective_model(
            self.disk_params, self.spf_params, self.psf_params, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel,
            stellar_psf_params=self.stellar_psf_params, StellarPSFModel=self.StellarPSFModel,
            throughput=self.throughput, **self.kwargs
        )
    
    def get_objective_likelihood(self, params_fit, fit_keys, target_image, err_map):
        """
        Returns the log likelihood of the disk model as per the given parameters, target image, and error map.

        Parameters
        ----------
        params_fit: list of float
            List of parameter values that are set to replace the original values of the parameters specified by fit_keys.
        fit_keys: list of str
            List of names of parameters that are set to be replaced by values in params_fit.
        target_image: numpy.ndarray
            Target image that the log likelihood is being computed with.
        err_map: numpy.ndarray
            Error map of same shape as target image that the log likelihood is being computed with.

        Returns:
        --------
        float
            Log likelihood value of the current model.
        """
        self._set_size_to_target_image(target_image)
        return objective_fit(
            params_fit, fit_keys, self.disk_params, self.spf_params, self.psf_params, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, target_image, err_map,
            stellar_psf_params=self.stellar_psf_params, StellarPSFModel=self.StellarPSFModel,
            throughput = self.throughput, **self.kwargs
        )
    
    def get_gradient(self, keys, target_image, err_map):
        """
        Returns the gradient of the disk model for the given target image and error map for the parameters specified in keys.
        Note: Log-scaling is not handled in this method. Some parameters may not work.

        Parameters
        ----------
        keys: list of str
            Names of parameters that the gradient is being computed for.
        target_image: numpy.ndarray
            Target image that the gradient is being computed with.
        err_map: numpy.ndarray
            Error map of same shape as target image that the gradient is being computed with.

        Returns:
        --------
        list
            Gradient of the specified values in keys. Gradients returned in the same order as keys.
        """
        return self._convert_raw_gradient_output_to_readable_output(keys, self._get_raw_gradient(target_image, err_map))
    
    def get_objective_gradient(self, params_fit, fit_keys, target_image, err_map):
        """
        Objective function for returning the gradient of the disk model for the given target image and error map for
        the log likelihood. Takes in a list of parameters and their new values and outputs their gradients. Note:
        Log-scaling is not handled in this method. Some parameters may not work

        Parameters
        ----------
        params_fit: list of float
            List of parameter values that are set to replace the original values of the parameters specified by fit_keys.
        fit_keys: list of str
            Names of parameters that the gradient is being computed for.
        target_image: numpy.ndarray
            Target image that the gradient is being computed with.
        err_map: numpy.ndarray
            Error map of same shape as target image that the gradient is being computed with.

        Returns:
        --------
        numpy.ndarray
            Gradient of the specified values in fit_keys. Gradients returned in the same order as fit_keys.
        """
        return self._convert_raw_gradient_output_to_1d_array(fit_keys, objective_fit_grad(params_fit, fit_keys, self.disk_params,
            self.spf_params, self.psf_params, self.misc_params, self.DiskModel, self.DistrModel, self.FuncModel,
            self.PSFModel, target_image, err_map, stellar_psf_params=self.stellar_psf_params, StellarPSFModel=self.StellarPSFModel,
            throughput=self.throughput, **self.kwargs
        ))
    
    def get_disk(self):
        """
        Returns the disk model as per the current parameters without the off-axis psf nor stellar psf. Note: will take
        longer to run the first time as it needs to be JIT compiled due to the changing of the component classes for
        the objective function call.

        Returns
        -------
        numpy.ndarray
            2d image of the generated disk model.
        """
        return objective_model(
            self.disk_params, self.spf_params, None, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel, None,
            stellar_psf_params=None, StellarPSFModel=None, throughput = self.throughput,
            **self.kwargs
        )
    
    def get_values(self, keys):
        """
        Retrieves the current values of the specified model parameters.

        Parameters
        ----------
        keys : list of str
            List of parameter names to retrieve from the disk, SPF, PSF, stellar PSF, or misc parameter dictionaries.

        Returns
        -------
        list
            List of parameter values in the same order as the keys. If a key is not found, `None` is returned for that key.
        """
        values = []

        for key in keys:
            if self.disk_params is not None and key in self.disk_params:
                values.append(self.disk_params[key])
            elif self.spf_params is not None and key in self.spf_params:
                values.append(self.spf_params[key])
            elif self.psf_params is not None and key in self.psf_params:
                values.append(self.psf_params[key])
            elif self.stellar_psf_params is not None and key in self.stellar_psf_params:
                values.append(self.stellar_psf_params[key])
            elif self.misc_params is not None and key in self.misc_params:
                values.append(self.misc_params[key])
            else:
                values.append(None)

        return values

    def log_likelihood_pos(self, target_image, err_map):
        """
        Returns the negative log-likelihood of the current model compared to the target image.

        Parameters
        ----------
        target_image : numpy.ndarray
            2D array representing the target image.

        err_map : numpy.ndarray
            2D array of the same shape as target_image, representing the error map.

        Returns
        -------
        float
            Negative log-likelihood of the model with respect to the given image and error map.
        """
        return -log_likelihood(self.get_model(), target_image, err_map)

    def log_likelihood(self, target_image, err_map):
        """
        Returns the log-likelihood of the current model compared to the target image.

        Parameters
        ----------
        target_image : numpy.ndarray
            2D array representing the target image.

        err_map : numpy.ndarray
            2D array of the same shape as target_image, representing the error map.

        Returns
        -------
        float
            Log-likelihood of the model with respect to the given image and error map.
        """
        return log_likelihood(self.get_model(), target_image, err_map)
    
    def define_reference_images(self, reference_images):
        """
        Sets the reference images to be used for constructing the stellar PSF.

        Parameters
        ----------
        reference_images : list of numpy.ndarray
            List of 2D reference images to use for constructing the stellar PSF.
        """
        StellarPSFReference.reference_images = reference_images

    def scipy_optimize(self, fit_keys, logscaled_params, array_params, target_image, err_map,
                       disp_soln=False, iters=500, method=None, use_grad = False, scale = 1.,
                       ftol=1e-40, gtol=1e-40, eps=1e-40, **kwargs): 
        """
        Runs unconstrained optimization using SciPy's `minimize` on the specified parameters to maximize the log-likelihood.
        Uses current parameter dictionary values.

        Parameters
        ----------
        fit_keys : list of str
            Names of parameters to fit.
        logscaled_params : list of str
            Subset of fit_keys that are log-scaled.
        array_params : list of str
            Subset of fit_keys that are array-valued parameters.
        target_image : numpy.ndarray
            Target image to fit the model to.
        err_map : numpy.ndarray
            Error map associated with the target image.
        disp_soln : bool, optional
            If True, prints the optimization result after completion. Default is False.
        iters : int, optional
            Maximum number of iterations. Default is 500.
        method : str or None, optional
            Optimization method to use (e.g., 'BFGS', 'L-BFGS-B'). If None, defaults to method chosen by `minimize`.
        use_grad : bool, optional
            Whether to use the analytical gradient. Default is False.
        scale : float, optional
            Scaling factor applied to the objective value and gradient. Default is 1.
        ftol : float, optional
            Tolerance for change in function value for convergence. Default is 1e-40.
        gtol : float, optional
            Tolerance for the norm of the projected gradient. Default is 1e-40.
        eps : float, optional
            Step size used for numerical approximation of the Jacobian (if `use_grad` is False). Default is 1e-40.
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        soln : scipy.optimize.OptimizeResult
            The result of the optimization. Includes optimized parameters, success status, and diagnostic information.
        """

        self._set_size_to_target_image(target_image)
        
        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        ll = lambda x: -self.get_objective_likelihood(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys,
                                       target_image, err_map) / scale
        
        ll_grad = lambda x: -self.get_objective_gradient(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys,
                                    self.disk_params, self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params,
                                    self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, self.StellarPSFModel,
                                    target_image, err_map) / scale
        
        ll_grad = lambda x: self._adjust_gradient_for_logscales( -self.get_objective_gradient(self._unflatten_params(x, fit_keys,
                                logscales, is_arrays), fit_keys, target_image, err_map) / scale, fit_keys, logscales, is_arrays, x)
        
        jac = ll_grad if use_grad else None
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        soln = minimize(ll, init_x, method=method, jac=jac, 
                        options={'disp': True, 'maxiter': iters, 'ftol': ftol, 'gtol': gtol, 'eps': eps})

        param_list = self._unflatten_params(soln.x, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        if disp_soln:
            print(soln)

        self.last_fit = 'scipyminimize'

        return soln
    
    def scipy_bounded_optimize(self, fit_keys, fit_bounds, logscaled_params, array_params, target_image, err_map,
                       disp_soln=False, iters=500, ftol=1e-12, gtol=1e-12, eps=1e-8, scale = 1.,
                       use_grad=False, **kwargs):
        """
        Runs bounded optimization using SciPy's `minimize` with the L-BFGS-B algorithm to maximize the log-likelihood.
        Uses current parameter dictionary values.

        Parameters
        ----------
        fit_keys : list of str
            Names of parameters to fit.
        fit_bounds : tuple of list of float
            Tuple (lower_bounds, upper_bounds) giving bounds for each parameter. Each bound can be scalar or array-like.
        logscaled_params : list of str
            Subset of fit_keys that are log-scaled.
        array_params : list of str
            Subset of fit_keys that are array-valued parameters.
        target_image : numpy.ndarray
            Target image to fit the model to.
        err_map : numpy.ndarray
            Error map associated with the target image.
        disp_soln : bool, optional
            If True, prints the optimization result after completion. Default is False.
        iters : int, optional
            Maximum number of iterations. Default is 500.
        ftol : float, optional
            Tolerance for change in function value for convergence. Default is 1e-12.
        gtol : float, optional
            Tolerance for the norm of the projected gradient. Default is 1e-12.
        eps : float, optional
            Step size used for numerical approximation of the Jacobian (if `use_grad` is False). Default is 1e-8.
        scale : float, optional
            Scaling factor applied to the objective value and gradient. Default is 1.
        use_grad : bool, optional
            Whether to use the analytical gradient. Default is False.
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        soln : scipy.optimize.OptimizeResult
            The result of the optimization. Includes optimized parameters, success status, and diagnostic information.
        """

        self._set_size_to_target_image(target_image)

        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        ll = lambda x: -self.get_objective_likelihood(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys,
                                       target_image, err_map) / scale
        
        ll_grad = lambda x: self._adjust_gradient_for_logscales( -self.get_objective_gradient(self._unflatten_params(x, fit_keys,
                                logscales, is_arrays), fit_keys, target_image, err_map) / scale, fit_keys, logscales, is_arrays, x)
        
        jac = ll_grad if use_grad else None
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        lower_bounds, upper_bounds = fit_bounds
        bounds = []
        i = 0
        for key, low, high in zip(fit_keys, lower_bounds, upper_bounds):
            low = np.atleast_1d(low)
            high = np.atleast_1d(high)
            if is_arrays[i]:
                for l, h in zip(low, high):
                    if logscales[i]:
                        bounds.append((np.log(l+1e-14), np.log(h)))
                    else:
                        bounds.append((low[0], high[0]))
            else:
                if logscales[i]:
                    bounds.append((np.log(low+1e-14), np.log(high)))
                else:
                    bounds.append((low[0], high[0]))
            i+=1

        soln = minimize(ll, init_x, method='L-BFGS-B', bounds=bounds, jac=jac,
                        options={'disp': True, 'maxiter': iters, 'ftol': ftol, 'gtol': gtol, 'eps': eps})
        
        param_list = self._unflatten_params(soln.x, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        if disp_soln:
            print(soln)

        self.last_fit = 'scipyboundminimize'

        if isinstance(self.FuncModel, InterpolatedUnivariateSpline_SPF):
            self.scale_spline_to_fixed_point(0, 1)

        return soln

    def mcmc(self, fit_keys, logscaled_params, array_params, target_image, err_map, BOUNDS, nwalkers=250, niter=250, burns=50, 
            continue_from=False, scale_objective_function_for_shape=False,**kwargs):
        """
        Runs Markov Chain Monte Carlo (MCMC) sampling using an emcee-based sampler to estimate 
        posterior distributions for model parameters based on the log-likelihood function. Uses
        current parameter dictionary values.

        Parameters
        ----------
        fit_keys : list of str
            Names of parameters to optimize using MCMC.
        logscaled_params : list of str
            Subset of `fit_keys` whose values are log-scaled.
        array_params : list of str
            Subset of `fit_keys` that correspond to array-valued parameters.
        target_image : numpy.ndarray
            2D array of the observed image data to compare against the model.
        err_map : numpy.ndarray
            2D array of the same shape as `target_image` representing per-pixel uncertainties.
        BOUNDS : tuple of (list, list)
            Tuple of (lower_bounds, upper_bounds) for each parameter in `fit_keys`. 
            Each bound can be a scalar or a list if the parameter is an array.
        nwalkers : int, optional
            Number of MCMC walkers. Default is 250.
        niter : int, optional
            Total number of MCMC steps per walker. Default is 250.
        burns : int, optional
            Number of burn-in steps to discard from each chain. Default is 50.
        continue_from : bool, optional
            If True, continues from the previous MCMC run stored in the backend. Default is False.
        scale_objective_function_for_shape : bool, optional
            If True, scales the log-likelihood by the number of pixels in `target_image` to normalize shape differences. Default is False.
        **kwargs : dict
            Additional arguments passed to the underlying `MCMC_model.run()` method (e.g., custom moves).

        Returns
        -------
        MCMC_model
            The fitted `MCMC_model` object containing the sampler chain, statistics, and results.

        Raises
        ------
        Exception
            If the initial guess is out of the specified parameter bounds.
        """

        self._set_size_to_target_image(target_image)

        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        scale = jnp.size(target_image) if scale_objective_function_for_shape else 1.
        
        ll = lambda x: self.get_objective_likelihood(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys,
                                     target_image, err_map)
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        lower_bounds, upper_bounds = BOUNDS
        i = 0
        bounds = []
        for key, low, high in zip(fit_keys, lower_bounds, upper_bounds):
            low = np.atleast_1d(low)
            high = np.atleast_1d(high)
            if is_arrays[i]:
                if logscales[i]:
                    for l, h in zip(low, high):
                        bounds.append((np.log(np.maximum(l, 1e-14)), np.log(h)))
                else:
                    for l, h in zip(low, high):
                        bounds.append((l, h))
            else:
                if logscales[i]:
                    bounds.append((np.log(np.maximum(low[0], 1e-14)), np.log(high[0])))
                else:
                    bounds.append((low[0], high[0]))
            i+=1

        # Flatten the bound arrays for comparison
        init_lb, init_ub = zip(*bounds)
        init_lb = np.array(init_lb)
        init_ub = np.array(init_ub)

        if not (np.all(init_x >= init_lb) and np.all(init_x <= init_ub)):
            init_param_list = self._unflatten_params(init_x, fit_keys, logscales, is_arrays)
            init_lb_list = self._unflatten_params(init_lb, fit_keys, logscales, is_arrays)
            init_ub_list = self._unflatten_params(init_ub, fit_keys, logscales, is_arrays)
            print("Initial mcmc parameters are out of bounds!")
            output_string = ""
            for i in range(0, len(init_param_list)):
                if(np.any(init_param_list[i] < init_lb_list[i]) or np.any(init_param_list[i] > init_ub_list[i])):
                    output_string += (f"{fit_keys[i]}: {init_param_list[i]}, ")
            print(output_string[0:-2])
            raise Exception("MCMC Initial Bounds Exception")

        mc_model = MCMC_model(ll, (init_lb, init_ub), self.name)
        mc_model.run(init_x, nconst=1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns, continue_from=continue_from, **kwargs)

        mc_soln = mc_model.get_theta_median()
        param_list = self._unflatten_params(mc_soln, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        self.last_fit = 'mcmc'

        # Unlogscale the internal sampler chain
        array_lengths = [len(self._get_param_value(k)) if k in array_params else 1 for k in fit_keys]
        mcmc_model = OptimizeUtils.unlogscale_mcmc_model(mc_model, fit_keys, logscaled_params, array_params, array_lengths)

        # Scale spline to (0, 1) if FuncModel is a spline
        if isinstance(self.FuncModel, InterpolatedUnivariateSpline_SPF):
            current_val = InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
                InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], 
                                                    InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)), 0)
            scale_factor = 1.0 / current_val if current_val != 0 else 1.0
            self.scale_spline_to_fixed_point(0, 1)
            
            OptimizeUtils.scale_spline_chains(mc_model, fit_keys, array_params, array_lengths, self.spf_params, scale_factor)

        mc_model.discard = burns

        return mc_model
    
    def inc_bound_knots(self, buffer = 0):
        """
        Applying inclination bounds to the knot values in order to improve fitting. Note: only use this if you have a good
        initial inclination guess.

        Parameters:
        -----------
        buffer : int
            Degrees of inclination to be added to the knot bounds for padding.

        Returns:
        --------
        dict
            Copy of spf parameters for easier access. Should usually ignore.
        """
        if(self.spf_params['num_knots'] <= 0):
            if(self.disk_params['sma'] < 50):
                self.spf_params['num_knots'] = 5
            else:
                self.spf_params['num_knots'] = 7
        self.spf_params['forwardscatt_bound'] = jnp.cos(jnp.deg2rad(90-self.disk_params['inclination']-buffer))
        self.spf_params['backscatt_bound'] = jnp.cos(jnp.deg2rad(90+self.disk_params['inclination']+buffer))
        return self.spf_params
    
    # Have to call this when using spline spfs
    def initialize_knots(self, target_image, dhg_params = [0.5, 0.5, 0.5]):
        """
        Initializes spline knots and flux scaling value to relatively close values to the target image. Need to run this
        before any jit_compile method is run due to those methods relying on initialized knots.

        Parameters:
        -----------
        target_image : numpy.ndarray
            2d numpy array of target image
        dhg_params : list
            Initial double henyey greenstein function values that the spline will be oriented to match for the initial
            guess.
        """

        self._set_size_to_target_image(target_image)

        ## Get a good scaling
        y, x = np.indices(target_image.shape)
        y -= self.misc_params['ny']//2
        x -= self.misc_params['nx']//2 
        rads = np.sqrt(x**2+y**2)
        mask = (rads > 12)

        self.spf_params['knot_values'] = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(dhg_params, InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params))

        init_image = self.get_model()

        if self.disk_params['inclination'] > 70: 
            knot_scale = 1.*np.nanpercentile(target_image[mask], 99) / (jnp.nanmax(init_image) + 1e-40)
        else: 
            knot_scale = 0.2*np.nanpercentile(target_image[mask], 99) / (jnp.nanmax(init_image) + 1e-40)
            
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * knot_scale

        #if self.FuncModel == FixedInterpolatedUnivariateSpline_SPF:
            #adjust_scale = 1.0 / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
                #InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
                #0.0)
            #self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
            #self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale
        #else:
        self.scale_spline_to_fixed_point(0, 1)

    def compute_stellar_psf_image(self):
        """
        Just returns an image of the stellar psf model.

        Returns:
        --------
        numpy.ndarray
            2 dimensional numpy array of stellar psf image
        """
        return self.StellarPSFModel.compute_stellar_psf_image(self.StellarPSFModel.pack_pars(self.stellar_psf_params),
                                                              self.misc_params['nx'], self.misc_params['ny'])
    
    def set_empirical_psf(self, image):
        """
        Sets the empirical psf image to the given image.
        
        Params:
        -------
        numpy.ndarray
            2 dimensional numpy array of the empirical psf image
        """

        EMP_PSF.img = image

    def scale_spline_to_fixed_point(self, cosphi, spline_val):
        """
        Only works for Interpolated Spline SPF Functions. Scales the spline by a single constant to match it with
        a single point. Usually scaled to (cosphi = 0, spline value = 1).
        """
        adjust_scale = spline_val / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
            cosphi)
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
        self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale

    def fix_all_nonphysical_params(self):
        """
        Catch-all method that bounds all parameters with non-physical parameters to their max or min values.
        """
        if issubclass(self.FuncModel, InterpolatedUnivariateSpline_SPF):
            self.spf_params['knot_values'] = np.where(self.spf_params['knot_values'] < 1e-8, 1e-8, self.spf_params['knot_values'])
        if self.disk_params['e']<0:
            self.disk_params['e'] = 0

    def print_params(self):
        """
        Prints parameters to console.
        """
        print("Disk Params: " + str(self.disk_params))
        print("SPF Params: " + str(self.spf_params))
        print("PSF Params: " + str(self.psf_params))
        print("Stellar PSF Params: " + str(self.stellar_psf_params))
        print("Misc Params: " + str(self.misc_params))

    def plot_spline(self, num_points=100):
        fig, ax = plt.subplots()

        x = np.linspace(-1, 1, num_points)
        knots = InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)
        spline = InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], knots)

        y = InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(spline, x)
        y_knots = InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(spline, knots)

        ax.set_title("Spline Fit")
        ax.plot(x, y, label="Spline curve")
        ax.scatter(knots, y_knots, color="red", label="Knots")
        ax.legend()

        return fig

    def get_flux_scale(self):
        """
        Returns flux scaling value.
            
        Returns:
        --------
        float
            flux scaling value
        """
        return self.misc_params['flux_scaling']

    def _flatten_params(self, fit_keys, logscales, is_arrays):
        """
        Flatten parameters into a 1D array for optimization.
        
        Parameters:
        -----------
        fit_keys : list
            List of parameter keys to be included in the flattened array.
        logscales : list
            List of boolean values indicating whether each parameter should be log-scaled.
            Must be the same length as fit_keys.
        is_arrays : list
            List of boolean values indicating whether each parameter is an array.
            Must be the same length as fit_keys.
            
        Returns:
        --------
        numpy.ndarray
            Flattened parameter array.
        """
            
        # Ensure lists are the same length as fit_keys
        if len(logscales) != len(fit_keys) or len(is_arrays) != len(fit_keys):
            raise ValueError("scales and is_arrays must have the same length as fit_keys")
        
        param_list = []
        for i, key in enumerate(fit_keys):
            # Get parameter from appropriate dictionary
            if isinstance(self.disk_params, dict) and key in self.disk_params:
                value = self.disk_params[key]
            elif isinstance(self.spf_params, dict) and key in self.spf_params:
                value = self.spf_params[key]
            elif isinstance(self.psf_params, dict) and key in self.psf_params:
                value = self.psf_params[key]
            elif isinstance(self.stellar_psf_params, dict) and key in self.stellar_psf_params:
                value = self.stellar_psf_params[key]
            elif isinstance(self.misc_params, dict) and key in self.misc_params:
                value = self.misc_params[key]
            else:
                raise ValueError(f"{key} not in any of the parameter dictionaries!")
                
            # Apply log scaling if needed
            if logscales[i]:
                if is_arrays[i]:
                    # Handle array parameters with log scaling
                    value = np.log(np.maximum(value, 1e-14))  # Ensure positive values for log
                else:
                    # Handle scalar parameters with log scaling
                    value = np.log(max(value, 1e-14))
                    
            param_list.append(value)
                
        return np.concatenate([np.atleast_1d(x) for x in param_list])
    
    def initialize_stellar_psf_weights(self, target_image, rcond=1e-8):
        """
        Initialize the weights for the stellar PSF reference images by solving
        a linear least squares problem to best match the target image.

        This method computes a weighted linear combination of the reference PSF
        images that best fits the target image in a least-squares sense. The resulting
        weights are stored in `self.stellar_psf_params['stellar_weights']`.

        Parameters
        ----------
        target_image : ndarray
            The 2D target image that the stellar PSF model should fit.
        rcond : float, optional
            Cut-off ratio for small singular values in the least squares solver.
            Values smaller than `rcond * largest_singular_value` are treated as zero.
            Default is 1e-8.

        Returns
        -------
        None
        """

        self._set_size_to_target_image(target_image)

        N = len(StellarPSFReference.reference_images)
        H, W = target_image.shape
        A = np.stack([img.flatten() for img in StellarPSFReference.reference_images], axis=1)  # shape: (H*W, N)
        y = target_image.flatten()  # shape: (H*W,)

        # Solve Aw = y using least squares
        weights, residuals, rank, s = np.linalg.lstsq(A, y, rcond=rcond)
        
        self.stellar_psf_params['stellar_weights'] = weights

    def _unflatten_params(self, flattened_params, fit_keys, logscales, is_arrays):
        """
        Convert flattened parameter array back to appropriate parameter values.
        
        Parameters:
        -----------
        flattened_params : numpy.ndarray
            Flattened parameter array.
        fit_keys : list
            List of parameter keys corresponding to the flattened parameters.
        logscales : list
            List of boolean values indicating whether each parameter should be log-scaled.
            Must be the same length as fit_keys.
        is_arrays : list
            List of boolean values indicating whether each parameter is an array.
            Must be the same length as fit_keys.
            
        Returns:
        --------
        list
            List of unflattened parameter values.
        """
            
        # Ensure lists are the same length as fit_keys
        if len(logscales) != len(fit_keys) or len(is_arrays) != len(fit_keys):
            raise ValueError("scales and is_arrays must have the same length as fit_keys")
        
        param_list = []
        index = 0
        
        for i, key in enumerate(fit_keys):
            if is_arrays[i]:
                # For arrays, determine the size
                for param_dict in [self.disk_params, self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params]:
                    if isinstance(param_dict, dict) and key in param_dict and hasattr(param_dict[key], "__len__"):
                        array_size = len(param_dict[key])
                        break
                else:
                    raise ValueError(f"Cannot determine array size for {key}")
                
                # Extract array values
                array_values = flattened_params[index:index+array_size]
                index += array_size
                
                # Apply inverse scaling if needed
                if logscales[i]:
                    array_values = np.exp(array_values)
                    
                param_list.append(array_values)
            else:
                # Handle scalar parameters
                value = flattened_params[index]
                index += 1
                
                # Apply inverse scaling if needed
                if logscales[i]:
                    value = np.exp(value)
                    
                param_list.append(value)
                
        return param_list

    def _update_params(self, param_values, fit_keys):
        """
        Update class parameter dictionaries with new values.
        
        Parameters:
        -----------
        param_values : list
            List of parameter values.
        fit_keys : list
            List of parameter keys corresponding to the parameter values.
        """
        if len(param_values) != len(fit_keys):
            raise ValueError("param_values must have the same length as fit_keys")
            
        for i, key in enumerate(fit_keys):
            value = param_values[i]
            
            if isinstance(self.disk_params, dict) and key in self.disk_params:
                self.disk_params[key] = value
            elif isinstance(self.spf_params, dict) and key in self.spf_params:
                self.spf_params[key] = value
            elif isinstance(self.psf_params, dict) and key in self.psf_params:
                self.psf_params[key] = value
            elif isinstance(self.stellar_psf_params, dict) and key in self.stellar_psf_params:
                self.stellar_psf_params[key] = value
            elif isinstance(self.misc_params, dict) and key in self.misc_params:
                self.misc_params[key] = value
            else:
                raise ValueError(f"{key} not in any of the parameter dictionaries!")
        
        self.fix_all_nonphysical_params()

    def _highlight_selected_params(self, fit_keys, selected_params):
        """
        Identify which parameters in `fit_keys` are also in `selected_params`.

        Parameters
        ----------
        fit_keys : list of str
            List of all parameter keys to check.
        selected_params : list of str
            Subset of keys that should be marked as selected (e.g., log-scaled or arrays).

        Returns
        -------
        list of bool
            Boolean list indicating whether each key in `fit_keys` is in `selected_params`.
        """

        select_bools = []
        for key in fit_keys:
            select_bools.append(key in selected_params)
        return select_bools

    def save_human_readable(self,dirname):
        with open(os.path.join(dirname,'{}_{}_hrparams.txt'.format(self.name,self.last_fit)), 'w') as save_file:
            save_file.write('Model Name: {}\n \n'.format(self.name))
            save_file.write('Method: {}\n \n'.format(self.last_fit))
            save_file.write('### Disk Params ### \n')
            for key in self.disk_params:
                save_file.write("{}: {}\n".format(key, self.disk_params[key]))
            save_file.write('\n### SPF Params ### \n')
            for key in self.spf_params:
                save_file.write("{}: {}\n".format(key, self.spf_params[key]))
            save_file.write('\n### PSF Params ### \n')
            for key in self.psf_params:
                save_file.write("{}: {}\n".format(key, self.psf_params[key]))
            save_file.write('\n### Misc Params ### \n')
            for key in self.misc_params:
                save_file.write("{}: {}\n".format(key, self.misc_params[key]))
        print("Saved human readable file to {}".format(os.path.join(dirname,'{}_{}_hrparams.txt'.format(self.name,self.last_fit))))

    def _get_param_value(self, key):
        """
        Retrieve the value of a parameter from one of the parameter dictionaries.

        Parameters
        ----------
        key : str
            Name of the parameter to retrieve.

        Returns
        -------
        Any
            The value of the parameter.

        Raises
        ------
        KeyError
            If the parameter is not found in any of the known parameter dictionaries.
        """

        param_dicts = [self.disk_params, self.spf_params, self.stellar_psf_params, self.misc_params]
        if isinstance(self.psf_params, dict):
            param_dicts.append(self.psf_params)
        for param_dict in param_dicts:
            if key in param_dict:
                return param_dict[key]
        raise KeyError(f"{key} not found in any parameter dict.")

    def save_machine_readable(self,dirname):
        with open(os.path.join(dirname,'{}_{}_diskparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            json.dump(self.disk_params, save_file)
        with open(os.path.join(dirname,'{}_{}_spfparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            serializable_spf = {}
            for key, value in self.spf_params.items():
                if isinstance(value, jnp.ndarray) or isinstance(value, np.ndarray):
                    serializable_spf[key] = value.tolist()
                else:
                    serializable_spf[key] = value
            json.dump(serializable_spf, save_file)
        with open(os.path.join(dirname,'{}_{}_psfparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            json.dump(self.psf_params, save_file)
        with open(os.path.join(dirname,'{}_{}_miscparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            serializable_misc = {}
            for key, value in self.misc_params.items():
                if isinstance(value, jnp.ndarray) or isinstance(value, np.ndarray):
                    serializable_misc[key] = value.tolist()
                else:
                    serializable_misc[key] = value
            json.dump(serializable_misc, save_file)
        print("Saved machine readable files to json in "+dirname)
    
    def load_machine_readable(self,dirname,method=None):
        ### defaults to last fitting mechanism, but can be changed to scipyminimize, scipyboundminimize, or mcmc
        if method == None:
            method = self.last_fit
        if self.last_fit == None:
            raise Exception("No last fit to load from. Please run a fit before loading.")
        else:
            try:
                with open(os.path.join(dirname,'{}_{}_diskparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    self.disk_params = json.load(read_file)
                with open(os.path.join(dirname,'{}_{}_spfparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    serializable_spf = json.load(read_file)
                    for key, value in serializable_spf.items():
                        if isinstance(value, list):
                            self.spf_params[key] = jnp.array(value)
                        else:
                            self.spf_params[key] = value
                with open(os.path.join(dirname,'{}_{}_psfparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    self.psf_params = json.load(read_file)
                with open(os.path.join(dirname,'{}_{}_miscparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    serializable_misc = json.load(read_file)
                    for key, value in serializable_misc.items():
                        if isinstance(value, list):
                            self.misc_params[key] = jnp.array(value)
                        else:
                            self.misc_params[key] = value
                print("Loaded machine readable files from json in "+dirname)
            except FileNotFoundError:
                print("File not found. Please check the directory and file names.")
                return
            
    def _convert_raw_gradient_output_to_readable_output(self, keys, gradients):
        """
        Convert raw gradient output into a list of gradients for the given parameter keys.

        Parameters
        ----------
        keys : list of str
            List of parameter keys for which to extract gradients.
        gradients : tuple
            Output from `_get_raw_gradient`, containing gradient arrays for different parameter groups.

        Returns
        -------
        list
            Gradient values corresponding to each parameter key in `keys`.

        Raises
        ------
        KeyError
            If a key is not found in any of the recognized gradient sources.
        """
        grad_disk = gradients[0]
        grad_spf = gradients[1]
        grad_psf = gradients[2] if len(gradients) > 2 else None
        grad_stellar = gradients[3] if len(gradients) > 3 else gradients[2]

        grads = []

        # Precompute index maps for efficiency
        disk_idx_map = {k: i for i, k in enumerate(self.disk_params)}
        spf_idx_map = {k: i for i, k in enumerate(self.spf_params)}
        psf_idx_map = {k: i for i, k in enumerate(self.psf_params)} if (self.PSFModel != Winnie_PSF and self.PSFModel != None) else {}
        
        use_interpolated_spf = issubclass(self.FuncModel, InterpolatedUnivariateSpline_SPF)
        use_winnie_psf = self.PSFModel != None and issubclass(self.PSFModel, Winnie_PSF)

        if self.StellarPSFModel is not None:
            stellar_grad_dict = self.StellarPSFModel.unpack_pars(grad_stellar)

        for key in keys:
            if key in disk_idx_map:
                grads.append(grad_disk[disk_idx_map[key]])
            elif key in spf_idx_map:
                if use_interpolated_spf:
                    grads.append(grad_spf)  # entire grad_spf array
                else:
                    grads.append(grad_spf[spf_idx_map[key]])
            elif not use_winnie_psf and key in psf_idx_map:
                grads.append(grad_psf[psf_idx_map[key]])
            elif self.StellarPSFModel is not None and key in self.stellar_psf_params:
                grads.append(jnp.ravel(stellar_grad_dict[key]).tolist())
            else:
                raise KeyError(f"Unrecognized key '{key}' in gradient.")

        return grads

    def _convert_raw_gradient_output_to_1d_array(self, keys, gradients):
        """
        Convert raw gradient output into a flattened 1D gradient array for optimization routines.

        Parameters
        ----------
        keys : list of str
            List of parameter keys for which to extract gradients.
        gradients : tuple
            Output from `_get_raw_gradient`, containing gradient arrays for different parameter groups.

        Returns
        -------
        ndarray
            1D array of gradient values for the specified parameter keys.

        Raises
        ------
        KeyError
            If a parameter key is not recognized.
        """

        grad_disk = gradients[0]
        grad_spf = gradients[1]
        grad_psf = gradients[2] if len(gradients) > 2 else None
        grad_stellar = gradients[3] if len(gradients) > 3 else gradients[2]

        grad_vector = []

        # Precompute lookup tables
        disk_idx_map = {k: i for i, k in enumerate(self.disk_params)}
        spf_idx_map = {k: i for i, k in enumerate(self.spf_params)}
        psf_idx_map = {k: i for i, k in enumerate(self.psf_params)} if (self.PSFModel != Winnie_PSF and self.PSFModel != None) else {}

        use_interpolated_spf = issubclass(self.FuncModel, InterpolatedUnivariateSpline_SPF)
        use_winnie_psf = self.PSFModel != None and issubclass(self.PSFModel, Winnie_PSF)

        if self.StellarPSFModel is not None:
            stellar_grad_dict = self.StellarPSFModel.unpack_pars(grad_stellar)

        for key in keys:
            if key in disk_idx_map:
                grad_vector.append(grad_disk[disk_idx_map[key]])
            elif key in spf_idx_map:
                if use_interpolated_spf:
                    grad_vector.extend(grad_spf.tolist())  # add all spline gradients
                else:
                    grad_vector.append(grad_spf[spf_idx_map[key]])
            elif not use_winnie_psf and key in psf_idx_map:
                grad_vector.append(grad_psf[psf_idx_map[key]])
            elif self.StellarPSFModel is not None and key in self.stellar_psf_params:
                grad_vector.extend(jnp.ravel(stellar_grad_dict[key]).tolist())  # stellar psf parameters are always arrays
            else:
                raise KeyError(f"Unrecognized key '{key}' in gradient conversion.")

        return np.array(grad_vector).flatten()
    
    def _get_raw_gradient(self, target_image, err_map):
        """
        Compute raw gradients with respect to all model parameters using `objective_grad`. Note: Logscaled parameters
        have their gradients with repect to their log values not their original values.

        Parameters
        ----------
        target_image : ndarray
            The target image to compare the model against.
        err_map : ndarray
            The error map for the image.

        Returns
        -------
        tuple
            Tuple containing raw gradients for (disk_params, spf_params, psf_params, stellar_psf_params).
        """
        self._set_size_to_target_image(target_image)
        return objective_grad(
            self.disk_params, self.spf_params, self.psf_params, self.misc_params, self.DiskModel,
            self.DistrModel, self.FuncModel, self.PSFModel, target_image, err_map,
            stellar_psf_params=self.stellar_psf_params, StellarPSFModel=self.StellarPSFModel, **self.kwargs
        )
    
    def _adjust_gradient_for_logscales(self, grad_list, fit_keys, logscales, is_arrays, flattened_params):
        """
        Adjust gradient values for parameters that are log-scaled using the chain rule.

        Parameters
        ----------
        grad_list : list or ndarray
            Raw gradient values before log-scale correction.
        fit_keys : list of str
            List of parameter keys being optimized.
        logscales : list of bool
            Boolean flags indicating which parameters are log-scaled.
        is_arrays : list of bool
            Boolean flags indicating which parameters are arrays.
        flattened_params : ndarray
            Flattened array of parameter values corresponding to `fit_keys`.

        Returns
        -------
        ndarray
            Corrected gradient array with chain rule applied to log-scaled parameters.

        Raises
        ------
        ValueError
            If the size of an array parameter cannot be determined from known dictionaries.
        """

        corrected = []
        grad_index = 0
        param_index = 0

        for i, key in enumerate(fit_keys):
            if is_arrays[i]:
                # Find the array size from the parameter dictionaries
                for param_dict in [self.disk_params, self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params]:
                    if isinstance(param_dict, dict) and key in param_dict and hasattr(param_dict[key], "__len__"):
                        array_size = len(param_dict[key])
                        break
                else:
                    raise ValueError(f"Cannot determine array size for {key}")

                grads = np.array(grad_list[grad_index:grad_index + array_size])
                if logscales[i]:
                    values = np.exp(flattened_params[param_index:param_index + array_size])
                    grads *= values  # Chain rule

                corrected.append(grads)

                grad_index += array_size
                param_index += array_size

            else:
                grad = grad_list[grad_index]
                if logscales[i]:
                    value = np.exp(flattened_params[param_index])
                    grad *= value  # Chain rule

                corrected.append(np.array([grad]))

                grad_index += 1
                param_index += 1

        return np.concatenate(corrected)
    
    def _set_size_to_target_image(self, target_image):
        """
        Set the model image size parameters to match the target image dimensions.

        Parameters
        ----------
        target_image : ndarray
            The target image whose dimensions will be used to set the model size.
        """
        self.misc_params['nx'] = target_image.shape[1]
        self.misc_params['ny'] = target_image.shape[0]

class OptimizeUtils:

    @classmethod
    def create_empirical_err_map(cls, data, annulus_width=5, mask_rad=9, outlier_pixels=None):
        """
        Constructs an empirical error map based on radial annuli standard deviation.

        Parameters
        ----------
        data : numpy.ndarray
            2D image data array.
        annulus_width : int, optional
            Width (in pixels) of each radial annulus. Default is 5.
        mask_rad : int, optional
            Radius (in pixels) within which the error is artificially set to a large value. Default is 9.
        outlier_pixels : list of tuple, optional
            List of pixel coordinates (tuples) to mark as extreme outliers with inflated error values.

        Returns
        -------
        noise_array : numpy.ndarray
            2D array of the same shape as `data`, representing the empirical error map.
        """
        y,x = np.indices(data.shape)
        y -= data.shape[0]//2
        x -= data.shape[1]//2 
        radii = np.sqrt(x**2 + y**2) 
        noise_array = np.zeros_like(data)
        for i in range(0, int(np.max(radii)//annulus_width) ): 
            indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
            noise_array[indices] = np.nanstd(data[indices])
        mask = radii <= mask_rad
        noise_array[mask] = 1e10

        if(outlier_pixels != None):
            for pixel in outlier_pixels:
                noise_array[pixel[0]][pixel[1]] = noise_array[pixel[0]][pixel[1]] * 1e6 

        return noise_array
    
    @classmethod
    def convert_dhg_params_to_spline_params(cls, g1, g2, w, spf_params):
        """
        Converts Double Henyey-Greenstein parameters into spline phase function values.

        Parameters
        ----------
        g1 : float
            First asymmetry parameter.
        g2 : float
            Second asymmetry parameter.
        w : float
            Weighting factor between the two lobes.
        spf_params : dict
            SPF parameters containing knot information for the spline.

        Returns
        -------
        spline_vals : numpy.ndarray
            Evaluated spline phase function values.
        """
        return DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([g1, g2, w], InterpolatedUnivariateSpline_SPF.get_knots(spf_params))

    @classmethod
    def process_image(cls, image, scale_factor=1, bounds=(70, 210, 70, 210)):
        """
        Crops, scales, and safely converts an image to float32.

        Parameters
        ----------
        image : numpy.ndarray
            Input image array.
        scale_factor : int, optional
            Factor to downsample the image spatially. Default is 1 (no scaling).
        bounds : tuple of int, optional
            A 4-element tuple defining the cropping window as (ymin, ymax, xmin, xmax). Default is (70, 210, 70, 210).

        Returns
        -------
        fin_image : numpy.ndarray
            Processed and cropped image array in float32 format with NaNs converted to 0.
        """
        cls.scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
        cropped_image = image[bounds[0]:bounds[1],bounds[0]:bounds[1]]
        def safe_float32_conversion(value):
            try:
                return np.float32(value)
            except (ValueError, TypeError):
                print("This value is unjaxable: " + str(value))
        fin_image = np.nan_to_num(cropped_image)
        fin_image = np.vectorize(safe_float32_conversion)(fin_image)
        return fin_image
    
    @classmethod
    def get_mask(cls, data, annulus_width=5, mask_rad=9, outlier_pixels=None):
        """
        Generates a binary mask identifying the central masked region of an image.

        Parameters
        ----------
        data : numpy.ndarray
            2D image array.
        annulus_width : int, optional
            Width of annuli used for noise estimation (not returned). Default is 5.
        mask_rad : int, optional
            Radius of the central region to mask out (set to True). Default is 9.
        outlier_pixels : list of tuple, optional
            Currently unused in mask generation but included for consistency.

        Returns
        -------
        mask : numpy.ndarray
            Boolean array of same shape as input, where True indicates masked (excluded) pixels.
        """
        y,x = np.indices(data.shape)
        y -= data.shape[0]//2
        x -= data.shape[1]//2 
        radii = np.sqrt(x**2 + y**2) 
        noise_array = np.zeros_like(data)
        for i in range(0, int(np.max(radii)//annulus_width) ): 
            indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
            noise_array[indices] = np.nanstd(data[indices])
        mask = radii <= mask_rad

        return mask
    
    @classmethod
    def unlogscale_mcmc_model(cls, mc_model, fit_keys, logscaled_params, array_params, array_lengths):
        """
        Unlog-scales MCMC chain values in-place for parameters originally fit in log space.

        Parameters
        ----------
        mc_model : MCMC_model
            Instance of MCMC_model whose chain will be modified.
        fit_keys : list of str
            Names of the parameters in the chain.
        logscaled_params : list of str
            Subset of fit_keys that were log-scaled during sampling.
        array_params : list of str
            Subset of fit_keys that are array-valued.
        array_lengths : list of int
            Length of each parameter in flattened form (1 for scalars, >1 for arrays).
        """
        chain = mc_model.sampler.get_chain()

        index = 0
        for i in range(len(fit_keys)):
            is_log = fit_keys[i] in logscaled_params
            is_array = fit_keys[i] in array_params
            length = array_lengths[i] if is_array else 1

            if is_log:
                chain[:, :, index:index+length] = np.exp(chain[:, :, index:index+length])
            index += length

        # Overwrite the sampler internals
        mc_model.scaled_chain = chain

    @classmethod
    def scale_spline_chains(cls, mc_model, fit_keys, array_params, array_lengths, scale_factor):
        """
        Applies normalization scaling to spline knots and compensates flux scaling if present in MCMC chains.

        Parameters
        ----------
        mc_model : MCMC_model
            The MCMC model object containing the chain.
        fit_keys : list of str
            Names of parameters in the MCMC chain.
        array_params : list of str
            Parameters treated as arrays (e.g., spline knots).
        array_lengths : list of int
            Flattened size of each parameter.
        scale_factor : float
            Value to scale `knot_values` by. If `flux_scaling` is present, it will be inversely scaled.
        """
        try:
            knot_idx = fit_keys.index('knot_values')
            if 'knot_values' not in array_params:
                return
        except ValueError:
            return
        
        start_idx = sum(array_lengths[:knot_idx])
        end_idx = start_idx + array_lengths[knot_idx]
        
        # Scale the chain
        chain = mc_model.sampler.get_chain().copy()

        chain[:, :, start_idx:end_idx] *= scale_factor
        
        # Scale flux_scaling inversely if being fit
        if 'flux_scaling' in fit_keys:
            flux_idx = sum(array_lengths[:fit_keys.index('flux_scaling')])
            chain[:, :, flux_idx] /= scale_factor

        mc_model.scaled_chain = chain