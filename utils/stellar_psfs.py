import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import jax.scipy.signal as jss

@register_pytree_node_class
class LinearStellar_PSF:
    def __init__(self, images):
        self.images = images

    def generate(self, image, weights):
        """
        Linearly combines the PSF images with the current weights.
        Returns a 2D image (same shape as individual images).
        """
        stacked = jnp.stack(self.images)  # shape (N, H, W)
        return jss.convolve2d(image, jnp.tensordot(weights, stacked, axes=1), mode='same') # shape (H, W)

    # PyTree methods
    def tree_flatten(self):
        # weights are dynamic (to be optimized), images are static
        children = ()
        aux_data = self.images
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        weights, = children
        return cls(aux_data)
