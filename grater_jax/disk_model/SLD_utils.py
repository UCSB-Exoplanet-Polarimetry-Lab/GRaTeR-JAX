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

        # Log-space computation to avoid overflow when alpha_in is large
        # and radial_ratio is small (e.g. (1e-8)^(-60) overflows float64).
        # Mathematically equivalent to:
        #   den = |rr|^(-2*alpha_in) + |rr|^(-2*alpha_out)
        #   radial_density = sqrt(2/den) * dens_at_r0
        log_rr = jnp.log(jnp.abs(radial_ratio) + 1e-8)
        log_den_in = -2.0 * distr["alpha_in"] * log_rr
        log_den_out = -2.0 * distr["alpha_out"] * log_rr
        log_den = jnp.logaddexp(log_den_in, log_den_out)
        radial_density_term = jnp.sqrt(2.0) * jnp.exp(-0.5 * log_den) * distr["dens_at_r0"]
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

        # Clamp denominator to prevent extreme gradients when g→±1
        # and cos_phi→±1 (near-forward/backward scattering singularity).
        denom = jnp.maximum(1 + p_dict["g"]**2 - 2*p_dict["g"]*cos_phi, 1e-8)
        return 1./(4*jnp.pi)*(1-p_dict["g"]**2) / (denom * jnp.sqrt(denom))


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

        # Clamp denominators to prevent extreme gradients when g→±1
        # and cos_phi→±1 (near-forward/backward scattering singularity).
        denom1 = jnp.maximum(1 + p_dict["g1"]**2 - 2*p_dict["g1"]*cos_phi, 1e-8)
        hg1 = p_dict['weight'] * 1./(4*jnp.pi)*(1-p_dict["g1"]**2) / \
            (denom1 * jnp.sqrt(denom1))
        denom2 = jnp.maximum(1 + p_dict["g2"]**2 - 2*p_dict["g2"]*cos_phi, 1e-8)
        hg2 = (1-p_dict['weight']) * 1./(4*jnp.pi)*(1-p_dict["g2"]**2) / \
            (denom2 * jnp.sqrt(denom2))

        return hg1+hg2
    

def recommended_num_knots(
    inclination_deg: float,
    num_knots_full_range: int,
    boundary_buffer: float = 0.1,
) -> int:
    """Return a recommended knot count scaled to the scattering angles probed at a given inclination.

    A disk at inclination *i* samples scattering angles roughly in the range
    ``[90 - i, 90 + i]`` degrees, which corresponds to cos(phi) in
    ``[-sin(i), +sin(i)]``.  Placing ``num_knots_full_range`` knots over the
    full 180-degree range is appropriate when the full range is probed (high
    inclinations), but over-parameterises the spline for low inclinations where
    only a narrow window is accessible.  Over-parameterisation makes the
    optimiser ill-conditioned and can cause convergence failures.

    This function scales the knot count linearly with the angular window that
    is actually probed, with a small outward buffer so knots are not placed
    right at the geometric boundary.

    The minimum returned value is 4.  While a cubic spline technically requires
    only 4 control points (``num_knots`` = 3), empirical testing shows that
    ``num_knots`` = 3 produces a spline that is too rigid: the optimiser
    consistently converges to image-consistent but SPF-inconsistent local
    minima at low inclinations.  ``num_knots`` = 4 (5 control points) provides
    enough flexibility to recover the SPF shape reliably.

    Using this function is optional.  Users can always pass ``num_knots``
    directly to ``InterpolatedUnivariateSpline_SPF.params``; this helper just
    provides a principled starting point when the inclination is known.

    Parameters
    ----------
    inclination_deg : float
        Disk inclination in degrees (0 = face-on, 90 = edge-on).
    num_knots_full_range : int
        Desired knot count for a disk that probes the full 0–180 degree range.
        Typically 5–8; the default ``InterpolatedUnivariateSpline_SPF`` uses 6.
    boundary_buffer : float, optional
        Buffer added beyond the geometric probed range in cos_phi units before
        computing the angular window.  Gives the spline a little room at the
        edges so knots are not placed exactly on the boundary.  Default 0.1.

    Returns
    -------
    int
        Recommended number of free knots, in the range
        ``[4, num_knots_full_range]``.

    Examples
    --------
    >>> recommended_num_knots(30.0, 6)
    4
    >>> recommended_num_knots(50.0, 6)
    4
    >>> recommended_num_knots(80.0, 6)
    6
    """
    sin_i = np.sin(np.radians(inclination_deg))
    cp_fwd  =  sin_i   # forward (small angle) boundary in cos_phi
    cp_back = -sin_i   # backward (large angle) boundary in cos_phi

    fwd_bound  = float(np.clip(cp_fwd  + boundary_buffer, -1.0, 1.0))
    back_bound = float(np.clip(cp_back - boundary_buffer, -1.0, 1.0))

    # Angular window in degrees — forward bound has larger cos_phi → smaller angle
    window_deg = float(
        np.degrees(np.arccos(back_bound)) - np.degrees(np.arccos(fwd_bound))
    )
    nk = max(4, round(num_knots_full_range * window_deg / 180.0))
    return int(nk)


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

    params = {'backscatt_bound': -1, 'forwardscatt_bound': 1, 'num_knots': 6, 'knot_values': (1., 1., 1., 1., 1., 1.)}

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def init(cls, p_arr, knots):
        """
        """
        return cls.pack_pars(p_arr, knots=knots)
    
    @classmethod
    def get_knots(cls, p_dict):
        """
        Return knot x-positions (in cos_phi space) for the spline SPF.

        Returns ``num_knots + 1`` positions.  A fixed knot is always placed at
        ``cos_phi = 0`` (90 degrees), which is the normalization point.  The
        remaining ``num_knots`` positions are split evenly on either side.
        """
        n = p_dict['num_knots']
        n_right = n // 2
        n_left = n - n_right
        left = jnp.linspace(p_dict['forwardscatt_bound'], 0.0, n_left + 1)[:-1]
        right = jnp.linspace(0.0, p_dict['backscatt_bound'], n_right + 1)[1:]
        return jnp.concatenate([left, jnp.array([0.0]), right])

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, p_arr, knots):
        """
        Build an InterpolatedUnivariateSpline from the free knot values.

        ``p_arr`` contains ``num_knots`` free values — the knot y-values at all
        positions *except* the fixed normalization point at ``cos_phi = 0``.  A
        value of 1.0 is inserted at the center index (``n_left = len(p_arr) // 2``)
        before constructing the spline, so ``spline(0) = 1`` by construction.
        This eliminates the degeneracy between knot scale and ``flux_scaling``.
        """
        n_left = len(p_arr) // 2
        all_values = jnp.concatenate([p_arr[:n_left], jnp.array([1.0]), p_arr[n_left:]])
        return InterpolatedUnivariateSpline(knots, all_values)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def compute_phase_function_from_cosphi(cls, spline_model, cos_phi):
        """
        Compute the phase function at (a) specific scattering scattering
        angle(s) phi. The argument is not phi but cos(phi) for optimization
        reasons.

        The spline is normalized so that it equals 1 at 90 degrees (cos_phi=0),
        breaking the degeneracy between the SPF knot values and the absolute
        flux scaling parameter.

        When the spline knots do not cover the full [-1, 1] cos_phi range
        (i.e., when inclination-dependent knot placement is used), values
        outside the knot range are extrapolated rather than evaluated with
        the boundary polynomial segment, which diverges for cubic splines:

        - **Forward side** (cos_phi > knot max): linear extrapolation using
          the spline's first derivative at the forward boundary. Physically
          motivated — SPFs typically rise toward forward scattering.
        - **Backward side** (cos_phi < knot min): constant extrapolation at
          the backward boundary value. SPFs are typically flat at large angles.

        When knots cover the full [-1, 1] range these branches are never
        triggered, so this is fully backward-compatible.

        Parameters
        ----------
        spline_model : InterpolatedUnivariateSpline
            spline model to represent scattering light phase function
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.
        """
        # Knots are stored in decreasing order: _x[0] is forward boundary
        # (largest cos_phi / smallest scattering angle) and _x[-1] is the
        # backward boundary (most negative cos_phi / largest angle).
        x_fwd  = spline_model._x[0]
        x_back = spline_model._x[-1]

        # In-range evaluation (clamp so the spline is never called out-of-range)
        cos_phi_clamped = jnp.clip(cos_phi, x_back, x_fwd)
        spline_val = spline_model(cos_phi_clamped)

        # Forward-side: linear extrapolation preserves the rising slope
        val_fwd   = spline_model(x_fwd)
        deriv_fwd = spline_model.derivative(x_fwd)
        fwd_extrap = val_fwd + deriv_fwd * (cos_phi - x_fwd)

        # Backward-side: constant extrapolation (SPF approximately flat there)
        back_extrap = spline_model(x_back)

        result = jnp.where(cos_phi > x_fwd,  fwd_extrap,
                 jnp.where(cos_phi < x_back, back_extrap, spline_val))

        norm = spline_model(jnp.array(0.0))
        norm = jnp.where(jnp.abs(norm) < 1e-10, 1.0, norm)
        return result / norm


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
        """Apply a Gaussian PSF to an image via FFT-based convolution.

        Parameters
        ----------
        image : jax.numpy.ndarray
            2D input image to be smoothed.
        psf_params : jax.numpy.ndarray
            Packed PSF parameter array as returned by ``pack_pars``.

        Returns
        -------
        jax.numpy.ndarray
            Smoothed image with the Gaussian PSF applied and offset added.
        """
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
        """Convolve the input image with the empirical PSF via FFT.

        Parameters
        ----------
        image : jax.numpy.ndarray
            2D input image to convolve.
        psf_params : jax.numpy.ndarray
            Packed PSF parameter array (unused; convolution uses ``cls.img``).

        Returns
        -------
        jax.numpy.ndarray
            Image convolved with the stored empirical PSF.
        """
        return jss.fftconvolve(image, cls.img, mode='same')

class Winnie_PSF(Jax_class):
    """
    Creates a JWST PSF model, using the package Winnie. See Winnie for further JWST PSF documentation.
    """
    @classmethod
    @partial(jax.jit, static_argnames=['cls', 'num_unique_psfs'])
    def init(cls, psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs):
        """Initialise and return a WinniePSF object from raw PSF grid arrays.

        Parameters
        ----------
        psfs : array-like
            PSF grid images.
        psf_inds_rolls : array-like
            Roll indices for each PSF.
        im_mask_rolls : array-like
            Image mask rolls.
        psf_offsets : array-like
            Positional offsets for each PSF.
        psf_parangs : array-like
            Position angles for each PSF.
        num_unique_psfs : int
            Number of unique PSFs in the grid.

        Returns
        -------
        WinniePSF
            Initialised Winnie PSF model object.
        """
        return WinniePSF(psfs, psf_inds_rolls, im_mask_rolls, psf_offsets, psf_parangs, num_unique_psfs)

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, winnie_model):
        """Return the WinniePSF model unchanged (identity packing).

        Parameters
        ----------
        winnie_model : WinniePSF
            Winnie PSF model object.

        Returns
        -------
        WinniePSF
            The same model object, unchanged.
        """
        return winnie_model

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, image, winnie_model):
        """Convolve an image with the Winnie PSF and return the mean over spacecraft rolls.

        Parameters
        ----------
        image : jax.numpy.ndarray
            2D input image to convolve.
        winnie_model : WinniePSF
            Packed Winnie PSF model as returned by ``pack_pars``.

        Returns
        -------
        jax.numpy.ndarray
            Mean of the roll-convolved image cube.
        """
        return jnp.mean(winnie_model.get_convolved_cube(image), axis=0)
    
class StellarPSFReference:

    """
    Reference images that the Stellar PSF classes will use.
    """

    reference_images = np.zeros((10, 10))

class LinearStellarPSF(Jax_class):
    """Stellar PSF model as a linear combination of reference images."""

    params = {'stellar_weights': None}  # Linear weights for each of the reference images.

    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def pack_pars(cls, p_dict):
        """Pack stellar PSF parameters into a flat array.

        Parameters
        ----------
        p_dict : dict
            Parameter dictionary with key ``'stellar_weights'``.

        Returns
        -------
        jax.numpy.ndarray
            1D array of stellar weights.
        """
        return p_dict['stellar_weights']

    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def unpack_pars(cls, stellar_psf_params):
        """Unpack a flat parameter array into the stellar PSF parameter dictionary.

        Parameters
        ----------
        stellar_psf_params : jax.numpy.ndarray
            1D array of stellar weights.

        Returns
        -------
        dict
            Dictionary with key ``'stellar_weights'``.
        """
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
        """Pack positional stellar PSF parameters into a single flat array.

        Concatenates ``stellar_weights``, ``stellar_xs``, and ``stellar_ys``
        in that order.

        Parameters
        ----------
        p_dict : dict
            Dictionary with keys ``'stellar_weights'``, ``'stellar_xs'``, and
            ``'stellar_ys'``.

        Returns
        -------
        jax.numpy.ndarray
            1D concatenated parameter array.
        """
        return jnp.concatenate([p_dict['stellar_weights'], p_dict['stellar_xs'], p_dict['stellar_ys']])
    
    @classmethod
    @partial(jax.jit, static_argnames=['cls'])
    def unpack_pars(cls, stellar_psf_params):
        """Unpack a flat parameter array into the positional stellar PSF dictionary.

        Splits the array into ``stellar_weights``, ``stellar_xs``, and
        ``stellar_ys`` based on the number of PSF reference images.

        Parameters
        ----------
        stellar_psf_params : jax.numpy.ndarray
            1D concatenated array produced by ``pack_pars``.

        Returns
        -------
        dict
            Dictionary with keys ``'stellar_weights'``, ``'stellar_xs'``,
            ``'stellar_ys'``.
        """
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
            """Bilinearly splat one weighted PSF reference onto the output pixel grid.

            Parameters
            ----------
            weight : float
                Scalar weight for this reference image.
            x : float
                Sub-pixel x position (row) of the PSF centre.
            y : float
                Sub-pixel y position (column) of the PSF centre.
            psf_img : jax.numpy.ndarray
                2D PSF reference image of shape (h, w).

            Returns
            -------
            tuple of jax.numpy.ndarray
                (all_x, all_y, all_v) — pixel row indices, column indices, and
                corresponding weighted values to scatter into the output image.
            """
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
                """Collect bilinear contributions for a single integer pixel shift.

                Parameters
                ----------
                shift : jax.numpy.ndarray
                    Length-2 integer array [dx, dy] representing the shift offset.
                weight_map : jax.numpy.ndarray
                    2D array of bilinear interpolation weights for this shift.

                Returns
                -------
                tuple of jax.numpy.ndarray
                    (xi, yi, val) — row indices, column indices, and weighted
                    values for valid (in-bounds) pixels only.
                """
                dx, dy = shift
                xi = x0i + dx  # (h, w)
                yi = y0i + dy
                val = weight * psf_img * weight_map

                xi = xi.reshape(-1)
                yi = yi.reshape(-1)
                val = val.reshape(-1)

                mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                val = jnp.where(mask, val, 0.0)
                xi = jnp.clip(xi, 0, nx - 1)
                yi = jnp.clip(yi, 0, ny - 1)

                return xi, yi, val

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
