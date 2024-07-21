import jax.scipy.optimize
from regression import log_likelihood_1d_pos_all_pars_spline, log_likelihood_1d_full_opt, log_likelihood_1d_pos_cent, log_likelihood,log_likelihood_1d_ecc_opt
from disk_utils_jax import jax_model_all_1d, jax_model_all_1d_cent, jax_model_all_1d_full
from scipy.optimize import minimize
import jaxopt
from SLD_utils import *
import jax.numpy as jnp


FULL_BOUNDS = [(1, 10), (-1, 10), (10, 80), (0, 180), (0, 180), (60, 80), (60, 80), (0, 10), (0, 10), (0, 10),
              (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5)]

def quick_optimize(target_image, err_map, flux_scaling=1e2, init_params = None, knots=jnp.linspace(1, -1, 6), grad = False, disp = True, method = None,
                    iters = 500, **kwargs):
    
    if(init_params == None):
        init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
        init_disk_guess = jnp.array([5., -5., 45., 45, 45])
        init_guess = jnp.concatenate([init_disk_guess, init_knot_guess])
        llp = lambda x: log_likelihood_1d_pos_all_pars_spline(x, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                            flux_scaling, target_image, err_map, knots=knots, **kwargs)
    else:
        init_guess = init_params

    grad_func = None
    if(grad):
        grad_func = jax.grad(llp)

    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, jac=grad_func)
    if(disp):
        print(soln)
    return soln.x

def quick_image(pars, flux_scaling=1e2, knots = jnp.linspace(1, -1, 6), **kwargs):
    return jax_model_all_1d(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0:5],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[5:], knots=knots), flux_scaling, **kwargs)

# 0: xc, 1: yc, 2: alpha_in, 3: alpha_out, 4: sma, 5: inclination, 6: position_angle
# 7 onwards is spline parameters, pxInArcsec and distance are good kwargs to include
def quick_optimize_cent(target_image, err_map, flux_scaling=1e2, init_params = None, knots=jnp.linspace(1, -1, 6), disp = True, method = None,
                    iters = 500, bounds = None, full_soln = False, **kwargs):

    if init_params == None:
        init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots)
        init_disk_guess = jnp.array([5., -5., 45., 45, 45])
        init_cent_guess = jnp.array([70., 70.])
        init_guess = jnp.concatenate([init_cent_guess, init_disk_guess, init_knot_guess])
    else:
        init_guess = init_params

    llp = lambda x: log_likelihood_1d_pos_cent(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, knots=knots, **kwargs)
    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, bounds=bounds,)
    if(disp):
        print(soln)
    if(full_soln):
        return soln
    return soln.x

@partial(jax.jit, static_argnums=(2))
def quick_image_cent(pars, flux_scaling=1e2, PSFModel = None, knots = jnp.linspace(1, -1, 6), **kwargs):
    return jax_model_all_1d_cent(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0], pars[1], pars[2:7],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[7:], knots=knots), flux_scaling, PSFModel=PSFModel, **kwargs)


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7:amin 8: ksi, 9: gamma, 10: beta
# 11 onwards is spline parameters, pxInArcsec and distance are good kwargs to include
def quick_optimize_full_opt(target_image, err_map, init_guess, flux_scaling=1e2, knots=jnp.linspace(1, -1, 6), jac = None, disp = True, method = None,
                    iters = 500, bounds = None, full_soln = False, **kwargs):

    llp = lambda x: log_likelihood_1d_full_opt(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, knots=knots, **kwargs)
    #soln, ignore = jaxopt.ScipyMinimize(fun=llp, method="bfgs", maxiter=200, jit=False).run(init_params=init_guess)
    #return soln
    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, jac=jac, bounds=bounds)
    if(disp):
        print(soln)
    if(full_soln):
        return soln
    return soln.x

def quick_image_full_opt(pars, flux_scaling=1e2, knots = jnp.linspace(1, -1, 6), **kwargs):
    return jax_model_all_1d_full(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0:11],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[11:], knots=knots), flux_scaling, **kwargs)

# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7:amin 8: ksi, 9: gamma, 10: beta, 11: e, 12: omega
# 13 onwards is spline parameters, pxInArcsec and distance are good kwargs to include
def quick_optimize_ecc_opt(target_image, err_map, init_guess, flux_scaling=1e2, knots=jnp.linspace(1, -1, 6), jac = None, disp = True, method = None,
                    iters = 500, bounds = None, full_soln = False, **kwargs):

    llp = lambda x: log_likelihood_1d_ecc_opt(x, 
                        DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, 
                        flux_scaling, target_image, err_map, knots=knots, **kwargs)
    #soln, ignore = jaxopt.ScipyMinimize(fun=llp, method="bfgs", maxiter=200, jit=False).run(init_params=init_guess)
    #return soln
    opt = {'disp':False,'maxiter':iters}
    soln = minimize(llp, init_guess, options=opt, method=method, jac=jac, bounds=bounds)
    if(disp):
        print(soln)
    if(full_soln):
        return soln
    return soln.x

def quick_image_ecc_opt(pars, flux_scaling=1e2, knots = jnp.linspace(1, -1, 6), **kwargs):
    return jax_model_all_1d_full(DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, pars[0:13],
                                InterpolatedUnivariateSpline_SPF.pack_pars(pars[13:], knots=knots), flux_scaling, **kwargs)