import jax
import jax.numpy as jnp

@jax.jit
def log_likelihood_pos_image(model_image, target_image, err_map):
    sigma2 = jnp.power(err_map, 2)
    result = jnp.power((target_image - model_image), 2) / sigma2 + jnp.log(sigma2)
    result = jnp.where(jnp.isnan(result), 0, result)
    return -0.5 * jnp.sum(result) #/ jnp.size(target_image)