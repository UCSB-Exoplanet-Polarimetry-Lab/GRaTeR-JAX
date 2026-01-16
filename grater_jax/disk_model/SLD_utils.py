"""
SLD_utils.py
============

Utility classes for disk modeling.

This module defines base JAX classes and implementations of density
distributions, scattering phase functions, and point spread functions
(PSFs) used in scattered-light disk forward modeling. It includes:

- `Jax_class` : Base class for packing/unpacking parameter dictionaries.
- `DustEllipticalDistribution2PowerLaws` : Two-power-law dust density model.
- `HenyeyGreenstein_SPF`, `DoubleHenyeyGreenstein_SPF` : Scattering phase functions.
- `InterpolatedUnivariateSpline_SPF` : Spline-based scattering phase function.
- `GAUSSIAN_PSF`, `EMP_PSF`, `Winnie_PSF` : PSF models.
- `LinearStellarPSF`, `PositionalStellarPSF` : Stellar PSF models using reference images.

This can be added to in order to introduce new distribution functions, scattering phase
functions, and point spread functions to the framework.
"""

import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from grater_jax.disk_model.interpolated_univariate_spline import InterpolatedUnivariateSpline
from astropy.io import fits
import jax.scipy.signal as jss
from grater_jax.disk_model.winnie_class import WinniePSF
import os

class Jax_class:
    """Base class for custom JAX-compatible objects that can be compressed into
    and uncompressed from JAX arrays."""

    params = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        Unpack a parameter array into a dictionary keyed by parameter names.

        Parameters
        ----------
        p_arr : jax.numpy.ndarray or array-like
            1D array of parameter values. The order must match the keys in
            ``cls.params``.

        Returns
        -------
        dict
            Dictionary mapping parameter names (str) to their corresponding values.
        """
        p_dict = {}
        keys = list(cls.params.keys())
        for i in range(len(p_arr)):
            p_dict[keys[i]] = p_arr[i]

        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        Pack a parameter dictionary into a JAX array.

        The order of parameters in the array is determined by the key order in
        ``cls.params``.

        Parameters
        ----------
        p_dict : dict
            Dictionary mapping parameter names (str) to values.

        Returns
        -------
        jax.numpy.ndarray
            1D array of parameter values in the order specified by ``cls.params``.
        """
        p_arrs = []
        for name in cls.params.keys():
            p_arrs.append(p_dict[name])
        return jnp.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """Two-power-law elliptical dust density distribution model."""

    params = {'alpha_in': 5., 'alpha_out': -5., 'sma': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'rmin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "rpeak": 0., "rpeak_surface_density": 0., "itiltthreshold": 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, accuracy=5.e-3, alpha_in=5., alpha_out=-5., sma=60., e=0., ksi0=1., gamma=2., beta=1., rmin=0., dens_at_r0=1.):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane.

        Based off of code from VIP disk forward modeling by Julien Milli

        Parameters
        ----------
        accuracy : float
            Density limit as described above. Default is 5.e-3.
        alpha_in : float
            slope of the power-low distribution in the inner disk. It must be positive (default 5)
        alpha_out : float
            slope of the power-low distribution in the outer disk. It must be negative (default -5)
        sma : float
            reference radius in au (default 60)
        e : float
            eccentricity (default 0)
        ksi0 : float
            scale height in au at the reference radius (default 1 a.u.)
        gamma : float
            exponent (2=gaussian,1=exponential profile, default 2)
        beta : float
            flaring index (0=no flaring, 1=linear flaring, default 1)
        rmin : float
            minimum semi-major axis: the dust density is 0 below this value (default 0)

        Other kwargs / params are generated from the above parameters.
        """

        p_dict = {}
        p_dict["accuracy"] = accuracy

        p_dict["ksi0"] = ksi0
        p_dict["gamma"] = gamma
        p_dict["beta"] = beta
        p_dict["zmax"] = ksi0*(-jnp.log(p_dict["accuracy"]))**(1./(gamma+1e-8))

        # Set Vertical Density Analogue
        gamma = jnp.where(gamma < 0., 0.1, gamma)
        ksi0 = jnp.where(ksi0 < 0., 0.1, ksi0)
        beta = jnp.where(beta < 0., 0., beta)

        # Set Radial Density Analogue
        alpha_in = jnp.where(alpha_in < 0.01, 0.01, alpha_in)
        alpha_out = jnp.where(alpha_out > -0.01, -0.01, alpha_out)
        e = jnp.where(e < 0., 0., e)
        e = jnp.where(e >= 1, 0.99, e)
        rmin = jnp.where(rmin < 0., 0., rmin)
        dens_at_r0 = jnp.where(dens_at_r0 < 0., 0., dens_at_r0)

        p_dict["alpha_in"] = alpha_in
        p_dict["alpha_out"] = alpha_out
        p_dict["sma"] = sma
        p_dict["e"] = e
        p_dict["p"] = p_dict["sma"]*(1-p_dict["e"]**2)
        p_dict["rmin"] = rmin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["rmin"]*(1-p_dict["e"]**2)
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["sma"]*p_dict["accuracy"]**(1/(p_dict["alpha_out"]+1e-8))
        p_dict["rpeak"] = p_dict["sma"] * jnp.power(-p_dict["alpha_in"]/(p_dict["alpha_out"]+1e-8),
                                        1./(2.*(p_dict["alpha_in"]-p_dict["alpha_out"])))
        Gamma_in = jnp.abs(p_dict["alpha_in"]+p_dict["beta"] + 1e-8)
        Gamma_out = -jnp.abs(p_dict["alpha_out"]+p_dict["beta"] + 1e-8)
        p_dict["rpeak_surface_density"] = p_dict["sma"] * jnp.power(-Gamma_in/Gamma_out,
                                                        1./(2.*(Gamma_in-Gamma_out+1e-8)))
        # the above formula comes from Augereau et al. 1999.
        p_dict["itiltthreshold"] = jnp.rad2deg(jnp.arctan(p_dict["rmax"]/p_dict["zmax"]))

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particule volume density at r, theta, z
        """
        distr = cls.unpack_pars(distr_params)

        radial_ratio = r*(1-distr["e"]*costheta)/((distr["p"])+1e-8)

        den = (jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["alpha_in"]) +
               jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["alpha_out"]))

        radial_density_term = jnp.sqrt(2./(den+1e-8))*distr["dens_at_r0"]
        #if distr["pmin"] > 0:
        #    radial_density_term[r/(distr["pmin"]/(1-distr["e"]*costheta)) <= 1] = 0
        radial_density_term = jnp.where(distr["pmin"] > 0, 
                                        jnp.where(r*(1-distr["e"]*costheta)/((distr["p"])+1e-8) <= 1, 0., radial_density_term),
                                        radial_density_term)

        den2 = distr["ksi0"]*jnp.power(jnp.abs(radial_ratio+1e-8), distr["beta"]) + 1e-8
        vertical_density_term = jnp.exp(-jnp.power((jnp.abs(z)+1e-8)/(jnp.abs(den2+1e-8)), jnp.abs(distr["gamma"])+1e-8))
        return radial_density_term*vertical_density_term

class HenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a single Henyey
    Greenstein function.
    """

    params = {'g': 0.3}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        Constructor of a Heyney Greenstein phase function.

        Parameters
        ----------
        spf_dico :  dictionary containing the key "g" (float)
            g is the Heyney Greenstein coefficient and should be between -1
            (backward scattering) and 1 (forward scattering).
        """

        p_dict = {}
        g = func_params[0]
        g = jnp.where(g>=1, 0.99, g)
        g = jnp.where(g<=-1, -0.99, g)
        p_dict["g"] = g

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        p_dict = cls.unpack_pars(phase_func_params)
        
        return 1./(4*jnp.pi)*(1-p_dict["g"]**2) / \
            (1+p_dict["g"]**2-2*p_dict["g"]*cos_phi)**(3./2.)


class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.

    Parameters
    ----------
    g1: float
        the first Heyney Greenstein coefficient and should be between -1
        (backward scattering) and 1 (forward scattering)
    g2: float
        the second Heyney Greenstein coefficient and should be between -1
        (backward scattering) and 1 (forward scattering)
    weight: float
        weighting of the first Henyey Greenstein component
    """

    params = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        """

        p_dict = {}
        p_dict['g1'] = func_params[0]
        p_dict['g2'] = func_params[1]
        p_dict['weight'] = func_params[2]

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def compute_phase_function_from_cosphi(cls, phase_func_params, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """

        p_dict = cls.unpack_pars(phase_func_params)

        hg1 = p_dict['weight'] * 1./(4*jnp.pi)*(1-p_dict["g1"]**2) / \
            (1+p_dict["g1"]**2-2*p_dict["g1"]*cos_phi)**(3./2.)
        hg2 = (1-p_dict['weight']) * 1./(4*jnp.pi)*(1-p_dict["g2"]**2) / \
            (1+p_dict["g2"]**2-2*p_dict["g2"]*cos_phi)**(3./2.)
        
        return hg1+hg2
    

# Uses 6 knots by default
# Values must be cos(phi) not phi
class InterpolatedUnivariateSpline_SPF(Jax_class):
    """
    Implementation of a spline scattering phase function. Uses 6 knots by default, takes knot y values as parameters.
    Locations are fixed to the given knots, pack_pars and init both return the spline model itself

    Parameters
    ----------
    backscatt_bound: float
        cosine of bound on back scattering (closer to 180 deg) scattering angle used for the spline
    forwardscatt_bound: float
        cosine of bound on forward scattering (closer to 0 deg) scattering angle used for the spline
    num_knots: int
        number of knots
    knot_values: array
        y values of the knots
    """

    params = {'backscatt_bound': -1, 'forwardscatt_bound': 1, 'num_knots': 6, 'knot_values': jnp.ones(6)}

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def init(cls, p_arr, knots):
        """
        """
        return cls.pack_pars(p_arr, knots=knots)
    
    @classmethod
    def get_knots(cls, p_dict):
        return jnp.linspace(p_dict['forwardscatt_bound'], p_dict['backscatt_bound'], p_dict['num_knots'])

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, p_arr, knots):
        """
        This function takes a array of (knots) values and converts them into an InterpolatedUnivariateSpline model.
        Also has inclination bounds which help narrow the spline fit
        """    
        return InterpolatedUnivariateSpline(knots, p_arr)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def compute_phase_function_from_cosphi(cls, spline_model, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        Parameters
        ----------
        spline_model : InterpolatedUnivariateSpline
            spline model to represent scattering light phase function
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        
        return spline_model(cos_phi)


class GAUSSIAN_PSF(Jax_class):

    """
    Gaussian PSF model. The PSF is defined by the following parameters:

    Parameters
    ----------
    FWHM : float
        Full width at half maximum of the Gaussian PSF.
    xo : float
        X coordinate of the center of the PSF.
    yo : float 
        Y coordinate of the center of the PSF.
    theta : float   
        Rotation angle of the PSF in radians.
    offset : float
        Offset value to be added to the PSF.
    amplitude : float
        Amplitude of the PSF.
    """

    params = {'FWHM': 3., 'xo': 0., 'yo': 0., 'theta': 0., 'offset': 0., 'amplitude': 1.}

    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, image, psf_params):
        ny, nx = image.shape    # Get image size
        p_dict = cls.unpack_pars(psf_params)
        FWHM = p_dict["FWHM"]
        amplitude = p_dict["amplitude"]
        offset = p_dict["offset"]
        theta = p_dict["theta"]
        sigma = FWHM / 2.355
        fx = jnp.fft.fftfreq(nx)  # cycles per pixel
        fy = jnp.fft.fftfreq(ny)
        FX, FY = jnp.meshgrid(fx, fy) # Rotating the frequency grid
        cost = jnp.cos(theta)
        sint = jnp.sin(theta)
        FXr = FX * cost + FY * sint
        FYr = -FX * sint + FY * cost
        gaussian_filter = jnp.exp(
            -2.0 * (jnp.pi ** 2) * (sigma ** 2) * (FXr ** 2 + FYr ** 2)
        )
        gaussian_filter = amplitude * gaussian_filter
        img_fft = jnp.fft.fft2(image)
        filtered_fft = img_fft * gaussian_filter
        smoothed = jnp.fft.ifft2(filtered_fft).real
        return smoothed + offset


class EMP_PSF(Jax_class):
    """Empirical point spread function (PSF) model."""

    params = {'scale_factor': 1.0, 'offset': 1.0}

    # Modify this to change the image the empirical psf uses
    img = None
    
    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, image, psf_params):
        return jss.fftconvolve(image, cls.img, mode='same')

class Winnie_PSF(Jax_class):
    """
    Creates a JWST PSF model, using the package Winnie. See Winnie for further JWST PSF documentation.
    """
    @classmethod
    @partial(jax.jit, static_argnames=['cls', 'num_unique_psfs'])
    def init(cls, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs):
        return WinniePSF(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs)

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, winnie_model):
        return winnie_model

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, image, winnie_model):
        return jnp.mean(winnie_model.get_convolved_cube(image), axis=0)
    
class StellarPSFReference:

    """
    Reference images that the Stellar PSF classes will use.
    """

    reference_images = jnp.zeros((10, 10))

class LinearStellarPSF(Jax_class):
    """Stellar PSF model as a linear combination of reference images."""

    params = {'stellar_weights': None}  # Linear weights for each of the reference images.

    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def pack_pars(cls, p_dict):
        return p_dict['stellar_weights']
    
    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def unpack_pars(cls, stellar_psf_params):
        p_dict = {}
        p_dict['stellar_weights'] = stellar_psf_params
        return p_dict

    @classmethod
    @partial(jax.jit, static_argnames=['cls', 'nx', 'ny'])
    def compute_stellar_psf_image(cls, stellar_weights, nx, ny):
        """
        Computes the on axis psf from the reference images and linear weights. Resizes the
        final image to (nx, ny).
        """
        image = jnp.tensordot(stellar_weights, StellarPSFReference.reference_images, axes=1)
        resized = jax.image.resize(image, (nx, ny), method='linear')
        return resized
    
class PositionalStellarPSF(Jax_class):
    """Stellar PSF model with position-dependent reference image weights."""

    params = {'stellar_weights': None, 'stellar_xs': None, 'stellar_ys': None}
    # Stellar weights : Linear weights for each of the reference images
    # Stellar xs and Stellar ys : X and Y positions for each of the reference images

    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def pack_pars(cls, p_dict):
        return jnp.concatenate([p_dict['stellar_weights'], p_dict['stellar_xs'], p_dict['stellar_ys']])
    
    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def unpack_pars(cls, stellar_psf_params):
        p_dict = {}
        psf_refs = StellarPSFReference.reference_images
        N, h, w = psf_refs.shape
        p_dict['stellar_weights'] = stellar_psf_params[0: N]
        p_dict['stellar_xs'] = stellar_psf_params[N: 2*N]
        p_dict['stellar_ys'] = stellar_psf_params[2*N: 3*N]
        return p_dict

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "nx", "ny"])
    def compute_stellar_psf_image(cls, stellar_psf_params, nx, ny):
        """
        Efficiently computes the resulting stellar psf from the linear weights,
        x positions, and y positions. Resizes the final image to (nx, ny).
        """
        psf_refs = StellarPSFReference.reference_images  # [N, h, w]
        N, h, w = psf_refs.shape
        p_dict = cls.unpack_pars(stellar_psf_params)

        xx = jnp.arange(h).reshape(h, 1)  # shape (h, 1)
        yy = jnp.arange(w).reshape(1, w)  # shape (1, w)

        def place_one(weight, x, y, psf_img):
            x0 = x - h / 2.0
            y0 = y - w / 2.0

            x_pix = x0 + xx  # shape (h, 1)
            y_pix = y0 + yy  # shape (1, w)

            x0f = jnp.floor(x_pix)
            y0f = jnp.floor(y_pix)

            dx = x_pix - x0f  # shape (h, 1)
            dy = y_pix - y0f  # shape (1, w)

            x0i = x0f.astype(jnp.int32)  # shape (h, 1)
            y0i = y0f.astype(jnp.int32)  # shape (1, w)

            # Broadcast to shape (h, w)
            dx = dx.repeat(w, axis=1)
            dy = dy.repeat(h, axis=0)
            x0i = x0i.repeat(w, axis=1)
            y0i = y0i.repeat(h, axis=0)

            # Bilinear weights
            w00 = (1 - dx) * (1 - dy)
            w10 = dx * (1 - dy)
            w01 = (1 - dx) * dy
            w11 = dx * dy

            shifts = jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]])
            weight_maps = jnp.stack([w00, w10, w01, w11], axis=0)

            def gather(shift, weight_map):
                dx, dy = shift
                xi = x0i + dx  # (h, w)
                yi = y0i + dy
                val = weight * psf_img * weight_map

                xi = xi.reshape(-1)
                yi = yi.reshape(-1)
                val = val.reshape(-1)

                mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                idx = jnp.nonzero(mask, size=xi.size, fill_value=0)[0]

                return xi[idx], yi[idx], val[idx]

            coords = [gather(shifts[i], weight_maps[i]) for i in range(4)]
            all_x = jnp.concatenate([c[0] for c in coords])
            all_y = jnp.concatenate([c[1] for c in coords])
            all_v = jnp.concatenate([c[2] for c in coords])
            return all_x, all_y, all_v

        x_list, y_list, v_list = jax.vmap(place_one, in_axes=(0, 0, 0, 0))(
            p_dict["stellar_weights"],
            p_dict["stellar_xs"],
            p_dict["stellar_ys"],
            psf_refs,
        )

        all_x = jnp.concatenate(x_list)
        all_y = jnp.concatenate(y_list)
        all_v = jnp.concatenate(v_list)

        acc = jnp.zeros((nx, ny))
        acc = acc.at[all_x, all_y].add(all_v)

        return acc
