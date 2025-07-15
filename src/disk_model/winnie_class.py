from functools import partial
import numpy as np
import webbpsf_ext
from disk_model.winnie_jwst_fm import generate_nircam_psf_grid
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import jax
from jax import vmap
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax.tree_util import register_pytree_node_class
from jax.image import resize
import jax.lax as lax

@register_pytree_node_class
class WinniePSF:

    def __init__(self, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs):
        self.psfs = jnp.array(psfs)
        self.psf_inds_rolls = jnp.array(psf_inds_rolls)
        self.im_mask_rolls = jnp.array(im_mask_rolls)
        self.psf_offsets = jnp.array(psf_offsets)
        self.psf_parangs = psf_parangs
        self.num_unique_psfs = num_unique_psfs

    def get_convolved_cube(self, image):
        convolved_cube = self.convolve_cube(image=image, cent=jnp.array((image.shape[1] / 2, image.shape[0] / 2)))
        return convolved_cube

    # Assuming parangs are already unique
    def convolve_cube(self, image, cent):
        nT = jnp.size(self.psf_parangs)
        ny, nx = image.shape[-2:]

        # Rotate the hypercube for each unique angle
        inp_rot_uni = self.rotate_hypercube(
            jnp.tile(image[None, ...], (nT, 1, 1)), self.psf_parangs, cent, cval0=0.
        )
        imcube = jnp.zeros((nT, ny, nx))

        # Convolve rotated images
        for i in range(0, jnp.size(self.psf_parangs)):
            imcube = imcube.at[i].set(
                self.convolve_with_spatial_psfs(
                    inp_rot_uni[i],
                    psf_inds=self.psf_inds_rolls[i],
                    im_mask=self.im_mask_rolls[i],
                )
            )
        return imcube

    def rotate_hypercube(self, hcube, angles, cent, cval0=0.):
        rot_hcube = jnp.empty_like(hcube)

        for i in range(0, jnp.size(angles)):
            rot_hcube = rot_hcube.at[i].set(jax_rotate_image(hcube[i], angles[i], reshape=False, cval=cval0, order=1))

        return rot_hcube
    

    def convolve_with_spatial_psfs(self, image, psf_inds, im_mask):
        
        im_mask = resize(im_mask, image.shape, method='linear')
        psf_inds = resize(psf_inds, image.shape, method='nearest')

        im = image * im_mask
        nonzero = im != 0
        psf_inds_masked = jnp.where(nonzero, psf_inds, -1)

        imcon = jnp.zeros_like(im)


        for i in jnp.unique(psf_inds_masked, size=self.num_unique_psfs)[1:]:    #### QUESTIONABLE: pre-computing the number of unique psfs
            mask_i = psf_inds_masked == i
            im_to_convolve = jnp.where(mask_i, im, 0.0)
            imcon += jax.scipy.signal.fftconvolve(im_to_convolve, self.psfs[i], mode='same')

        return imcon
    

    # Using pytrees for flattening/unflattening instead of manual flattening/unflattening from Jax_class
    def tree_flatten(self):
        children = (self.psfs, self.psf_inds_rolls, self.im_mask_rolls, self.psf_offsets, self.psf_parangs)
        aux_data = (self.num_unique_psfs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs = children
        num_unique_psfs = aux_data
        obj = cls(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs)
        return obj


def jax_rotate_image(image, angle, reshape=False, cval=0.0, order=1):
    
    angle_rad = jnp.deg2rad(angle)
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)

    # Compute the center of the image
    ny, nx = image.shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Generate coordinate grid
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')

    # Shift coordinates to center
    x_shifted = x - center_x
    y_shifted = y - center_y

    # Apply rotation matrix
    x_rot = cos_theta * x_shifted + sin_theta * y_shifted + center_x
    y_rot = -sin_theta * x_shifted + cos_theta * y_shifted + center_y

    # Interpolate using map_coordinates
    rotated_image = map_coordinates(image, [y_rot.ravel(), x_rot.ravel()], order=order, cval=cval)
    rotated_image = rotated_image.reshape(image.shape)

    return rotated_image