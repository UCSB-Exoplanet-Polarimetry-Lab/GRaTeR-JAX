"""
winnie_class.py
===============

JAX wrapper for spatially-varying JWST PSF convolution.

This module defines the `WinniePSF` class, a JAX-compatible implementation of
the Winnie package (Lawson et al.) for PSF subtraction. It provides tools for
constructing and applying spatially-varying PSFs across roll angles, including
rotation, masking, and convolution. A helper function `jax_rotate_image` is also
included for JAX-based image rotation.
"""

from functools import partial
import numpy as np
from grater_jax.disk_model.winnie_jwst_fm import generate_nircam_psf_grid
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
    """
    A class for spatially-varying PSF convolution with image cubes, including rotation
    and masking per roll angle. Class is a JAX pytree implementation of Kellen Lawson's
    incredible Winnie package for PSF subtraction.

    Parameters
    ----------
    psfs : array_like
        Array of PSF kernels, shape (N, H, W) where N is number of unique PSFs.
    psf_inds_rolls : array_like
        Array of PSF index maps for each roll angle. Shape (n_rolls, H, W).
    im_mask_rolls : array_like
        Array of image masks for each roll angle. Shape (n_rolls, H, W).
    psf_offsets : array_like
        Array of PSF offsets per roll (currently unused).
    psf_parangs : array_like
        Array of parallactic angles (in degrees) for each roll.
    num_unique_psfs : int
        The number of unique PSF kernels used across the image.

    Attributes
    ----------
    psfs : jax.numpy.ndarray
    psf_inds_rolls : jax.numpy.ndarray
    im_mask_rolls : jax.numpy.ndarray
    psf_offsets : jax.numpy.ndarray
    psf_parangs : jax.numpy.ndarray
    num_unique_psfs : int
    """

    def __init__(self, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs):
        self.psfs = jnp.array(psfs)
        self.psf_inds_rolls = jnp.array(psf_inds_rolls)
        self.im_mask_rolls = jnp.array(im_mask_rolls)
        self.psf_offsets = jnp.array(psf_offsets)
        self.psf_parangs = psf_parangs
        self.num_unique_psfs = num_unique_psfs

    @classmethod
    def init_db(cls, db, roll_index=0, 
                 normalize=True, log_rscale=True,
                 nr=5, rmax=10**0.5, ntheta=4, use_coeff=True):
        """
        Generate PSF grid inputs from a spaceKLIP Database and store in a form usable
        by a WinniePSF class.

        Parameters
        ----------
        db : spaceKLIP.database.Database
            Loaded spaceKLIP database object.
        roll_index : int, optional
            Index of roll angle to use for instrument setup (default is 0).
        normalize : bool
            Whether to normalize each PSF slice.
        log_rscale : bool
            Whether to sample PSFs logarithmically in radius.
        nr : int
            Number of radial zones.
        rmax : float
            Maximum radius (arcsec).
        ntheta : int
            Number of angular zones.
        use_coeff : bool
            Whether to use precomputed PSF coefficients.
        """
        # 1. Extract star centers and parallactic angles
        c_coron_rolls = np.array(db.star_centers)
        psf_parangs = np.array(db.roll_angles)  # in degrees

        # 2. Get the webbpsf instrument for the selected roll
        inst = db.get_webbpsf_instrument(roll_index)

        # 3. Generate PSF grid
        psfs, psf_inds_rolls, im_mask_rolls, psf_offsets = generate_nircam_psf_grid(
            cls.inst,
            cls.c_coron_rolls,
            source_spectrum=None,
            normalize=normalize,
            log_rscale=log_rscale,
            nr=nr,
            rmax=rmax,
            ntheta=ntheta,
            use_coeff=use_coeff
        )

        # 4. Store results as JAX arrays (optional, depending on use)
        psfs = jnp.array(psfs)
        psf_inds_rolls = jnp.array(psf_inds_rolls)
        im_mask_rolls = jnp.array(im_mask_rolls)
        psf_offsets = jnp.array(psf_offsets)
        n_unique_inds = len(jnp.unique(psf_inds_rolls))

        return cls(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, n_unique_inds)


    def get_convolved_cube(self, image):
        """
        Apply spatially-varying convolution to an input image across roll angles.

        Parameters
        ----------
        image : jax.numpy.ndarray
            2D input image to convolve.

        Returns
        -------
        jax.numpy.ndarray
            Convolved image cube of shape (n_rolls, H, W).
        """
        convolved_cube = self.convolve_cube(image=image, cent=jnp.array((image.shape[1] / 2, image.shape[0] / 2)))
        return convolved_cube

    def convolve_cube(self, image, cent):
        """
        Rotate and convolve the input image for each roll angle.

        Parameters
        ----------
        image : jax.numpy.ndarray
            Input 2D image.
        cent : tuple of float
            Center of rotation (x, y) in pixels.

        Returns
        -------
        jax.numpy.ndarray
            Image cube after convolution, shape (n_rolls, H, W).
        """
        nT = jnp.size(self.psf_parangs)
        ny, nx = image.shape[-2:]
        inp_rot_uni = self.rotate_hypercube(
            jnp.tile(image[None, ...], (nT, 1, 1)), self.psf_parangs, cent, cval0=0.
        )
        imcube = jnp.zeros((nT, ny, nx))

        for i in range(0, nT):
            imcube = imcube.at[i].set(
                self.convolve_with_spatial_psfs(
                    inp_rot_uni[i],
                    psf_inds=self.psf_inds_rolls[i],
                    im_mask=self.im_mask_rolls[i],
                )
            )
        return imcube

    def rotate_hypercube(self, hcube, angles, cent, cval0=0.):
        """
        Rotate each image slice in a cube by a corresponding angle.

        Parameters
        ----------
        hcube : jax.numpy.ndarray
            Input hypercube of shape (n_slices, H, W).
        angles : jax.numpy.ndarray
            Array of angles (degrees) for each slice.
        cent : tuple of float
            Center of rotation (x, y).
        cval0 : float, optional
            Fill value for outside edges (default is 0).

        Returns
        -------
        jax.numpy.ndarray
            Rotated hypercube with the same shape as input.
        """
        rot_hcube = jnp.empty_like(hcube)
        for i in range(0, jnp.size(angles)):
            rot_hcube = rot_hcube.at[i].set(
                jax_rotate_image(hcube[i], angles[i], reshape=False, cval=cval0, order=1)
            )
        return rot_hcube

    def convolve_with_spatial_psfs(self, image, psf_inds, im_mask):
        """
        Convolve an image using spatially-varying PSFs based on a mask and index map.

        Parameters
        ----------
        image : jax.numpy.ndarray
            The image to convolve.
        psf_inds : jax.numpy.ndarray
            Index map indicating which PSF to use at each pixel.
        im_mask : jax.numpy.ndarray
            Binary mask to apply before convolution.

        Returns
        -------
        jax.numpy.ndarray
            Convolved image.
        """
        im_mask = resize(im_mask, image.shape, method='linear')
        psf_inds = resize(psf_inds, image.shape, method='nearest')
        im = image * im_mask
        nonzero = im != 0
        psf_inds_masked = jnp.where(nonzero, psf_inds, -1)
        imcon = jnp.zeros_like(im)

        for i in jnp.unique(psf_inds_masked, size=self.num_unique_psfs)[1:]:
            mask_i = psf_inds_masked == i
            im_to_convolve = jnp.where(mask_i, im, 0.0)
            imcon += jax.scipy.signal.fftconvolve(im_to_convolve, self.psfs[i], mode='same')

        return imcon

    def tree_flatten(self):
        """
        Flatten the object for use with JAX PyTrees.

        Returns
        -------
        tuple
            A tuple (children, aux_data) where children are JAX-traceable arrays
            and aux_data is any other metadata.
        """
        children = (self.psfs, self.psf_inds_rolls, self.im_mask_rolls, self.psf_offsets, self.psf_parangs)
        aux_data = (self.num_unique_psfs)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten the object from children and auxiliary data.

        Parameters
        ----------
        aux_data : tuple
            Metadata (e.g. number of PSFs).
        children : tuple
            Flattened arrays representing object state.

        Returns
        -------
        WinniePSF
            Reconstructed object.
        """
        psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs = children
        num_unique_psfs = aux_data
        obj = cls(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs)
        return obj


def jax_rotate_image(image, angle, reshape=False, cval=0.0, order=1):
    """
    Rotate a 2D image using JAX-compatible coordinate transformations.

    Parameters
    ----------
    image : jax.numpy.ndarray
        Input 2D image.
    angle : float
        Rotation angle in degrees.
    reshape : bool, optional
        Whether to reshape output to fit the entire rotated image (not used).
    cval : float, optional
        Constant value for points outside boundaries.
    order : int, optional
        Spline interpolation order (0=nearest, 1=linear, etc).

    Returns
    -------
    jax.numpy.ndarray
        Rotated image of the same shape.
    """
    angle_rad = jnp.deg2rad(angle)
    cos_theta = jnp.cos(angle_rad)
    sin_theta = jnp.sin(angle_rad)
    ny, nx = image.shape
    center_y, center_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    y, x = jnp.meshgrid(jnp.arange(ny), jnp.arange(nx), indexing='ij')
    x_shifted = x - center_x
    y_shifted = y - center_y
    x_rot = cos_theta * x_shifted + sin_theta * y_shifted + center_x
    y_rot = -sin_theta * x_shifted + cos_theta * y_shifted + center_y
    rotated_image = map_coordinates(image, [y_rot.ravel(), x_rot.ravel()], order=order, cval=cval)
    return rotated_image.reshape(image.shape)
