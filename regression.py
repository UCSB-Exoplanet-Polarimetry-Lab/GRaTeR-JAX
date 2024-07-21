import jax
import jax.numpy as jnp
import numpy as np
from disk_utils_jax import jax_model, jax_model_1d, jax_model_all_1d, jax_model_all_1d_cent, jax_model_all_1d_full, jax_model_all_1d_ecc
from functools import partial


# Computes the error between the model_image and target_image
@jax.jit
def log_likelihood_image(model_image, target_image, err_map):
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)/jnp.size(target_image)

# Computes the error between the target_image and a disk generated from the parameters
@partial(jax.jit, static_argnums=(0,1,6))
def log_likelihood(DistrModel, FuncModel, disk_params, spf_params, target_image, err_map, PSFModel = None, **kwargs):
    model_image = jax_model(DistrModel, FuncModel, disk_params=disk_params,
                            PSFModel=PSFModel, spf_params=spf_params, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_params is a jax array)
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map, PSFModel = None, **kwargs):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                               PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_params is a jax array)
# Returns a positive number instead of negative number for future use
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d_pos(disk_params, DistrModel, FuncModel, spf_params, flux_scaling, target_image, err_map, PSFModel = None, **kwargs):
    model_image = jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                               PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_and_spf_params is a jax array
# and a combination of a disk_params array and an spf_param array) (first 6 values are disk's, rest are spf's)
# Returns a positive number instead of negative number for future use
# This method does not work with spline spfs
@partial(jax.jit, static_argnums=(1,2,6))
def log_likelihood_1d_pos_all_pars(disk_and_spf_params, DistrModel, FuncModel, flux_scaling, target_image, err_map, PSFModel = None, **kwargs):
    model_image = jax_model_all_1d(DistrModel, FuncModel, disk_and_spf_params[0:5], disk_and_spf_params[5:], flux_scaling,
                                    PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2) 
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (disk_and_spf_params is a jax array
# and a combination of a disk_params array and an spf_param array) (first 6 values are disk's, rest are spf's)
# Returns a positive number instead of negative number for future use
# This method is exclusively for spline spfs
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d_pos_all_pars_spline(disk_and_spf_params, DistrModel, FuncModel, flux_scaling, target_image, err_map,
                                            knots = jnp.linspace(1,-1,6), PSFModel = None, **kwargs):
    model_image = jax_model_all_1d(DistrModel, FuncModel, disk_and_spf_params[0:5], FuncModel.pack_pars(disk_and_spf_params[5:],
                                    knots=knots), flux_scaling, PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)

# Computes the error between the target_image and a disk generated from the parameters (both disk_params
# and spf_params are jax arrays)
# Returns a positive number instead of negative number for future use
# This method is exclusively for spline spfs
@partial(jax.jit, static_argnums=(2,3,8))
def log_likelihood_1d_pos_spline(disk_params, spf_params, DistrModel, FuncModel, flux_scaling, target_image, err_map,
                                            knots = jnp.linspace(1,-1,6), PSFModel = None, **kwargs):
    model_image = jax_model_all_1d(DistrModel, FuncModel, disk_params, FuncModel.pack_pars(spf_params, knots=knots), flux_scaling,
                                    PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)


# Basically log_likelihood_1d_pos_all_pars_spline but including cent as the first two values in the parameter array
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d_pos_cent(params, DistrModel, FuncModel, flux_scaling, target_image, err_map,
                                            knots = jnp.linspace(1,-1,6), PSFModel = None, **kwargs):
    model_image = jax_model_all_1d_cent(DistrModel, FuncModel, params[0], params[1], params[2:7], FuncModel.pack_pars(params[7:],
                                    knots=knots), flux_scaling, PSFModel=PSFModel, **kwargs)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)/jnp.size(target_image)


# Computes the error between the target_image and a disk generated from the parameters (all_params is a jax array)
# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: e, 8: ksi, 9: gamma, 10: beta
# 11 onwards is spline parameters
# pxInArcsec and distance are important kwargs to include
# Returns a positive number instead of negative number for future use
# This method is exclusively for spline spfs
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d_full_opt(all_params, DistrModel, FuncModel, flux_scaling, target_image, err_map,
                                               knots = jnp.linspace(1,-1,6), PSFModel = None, **kwargs):
    model_image = jax_model_all_1d_full(DistrModel, FuncModel, all_params[0:11], 
                                        FuncModel.pack_pars(all_params[11:], knots=knots), flux_scaling, PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.where(err_map > 0, jnp.power((target_image - model_image), 2) / (sigma2+1e-8) + jnp.log(sigma2+1e-8), 0)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)/jnp.size(target_image)

# Computes the error between the target_image and a disk generated from the parameters (all_params is a jax array)
# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: e, 8: ksi, 9: gamma, 10: beta
# 11 onwards is spline parameters
# pxInArcsec and distance are important kwargs to include
# Returns a positive number instead of negative number for future use
# This method is exclusively for spline spfs
@partial(jax.jit, static_argnums=(1,2,7))
def log_likelihood_1d_ecc_opt(all_params, DistrModel, FuncModel, flux_scaling, target_image, err_map,
                                               knots = jnp.linspace(1,-1,6), PSFModel = None, **kwargs):
    model_image = jax_model_all_1d_ecc(DistrModel, FuncModel, all_params[0:13], 
                                        FuncModel.pack_pars(all_params[13:], knots=knots), flux_scaling, PSFModel=PSFModel, **kwargs) # (y)
    sigma2 = jnp.power(err_map, 2)
    result = jnp.where(err_map > 0, jnp.power((target_image - model_image), 2) / (sigma2+1e-8) + jnp.log(sigma2+1e-8), 0)
    result = jnp.where(jnp.isnan(result), 0, result)
    return 0.5 * jnp.sum(result)/jnp.size(target_image)