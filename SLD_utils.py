import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from interpolated_univariate_spline import InterpolatedUnivariateSpline
from astropy.io import fits

# This file contains the Dust Distribution models, SPF models, and the helper Jax_class that makes everything run

class Jax_class:
    """Helper functions for JAX-related operations
    
    Methods
    ----------
    unpack_pars
    pack_pars
    """
    params = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array (params) and unpacks it into a
        dictionary with the parameter names as keys.

        Parameters
        -----------
        p_arr : array
            input parameter array
        
        Returns
        -----------
        p_dict : dict
            parameter dictionary
        """
        p_dict = {}
        keys = list(cls.params.keys())
        i = 0
        for i in range(0, len(p_arr)):
            p_dict[keys[i]] = p_arr[i]

        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.

        Parameters
        -----------
        p_dict : dict
            parameter dictionary

        Returns
        -----------
        jnp.asarray(p_arrs): jax array
            jax array of parameters
        """    
        p_arrs = []
        for name in cls.params.keys():
            p_arrs.append(p_dict[name])
        return jnp.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """ Creates a dust distribution that follows two power laws, following the parameters input in the JAX class; 
    referred to in other files as distr_cls with its object as DistrModel and parameters as distr_params
    
    Methods
    ----------
    density_cylindrical
    """

    params = {'ain': 5., 'aout': -5., 'a': 60., 'e': 0., 'ksi0': 1.,'gamma': 2., 'beta': 1.,
                        'amin': 0., 'dens_at_r0': 1., 'accuracy': 5.e-3, 'zmax': 0., "p": 0., "rmax": 0.,
                        'pmin': 0., "apeak": 0., "apeak_surface_density": 0., "itiltthreshold": 0.}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, accuracy=5.e-3, ain=5., aout=-5., a=60., e=0., ksi0=1., gamma=2., beta=1., amin=0., dens_at_r0=1.):
        """
        Constructor for the Dust_distribution class.

        We assume the dust density is 0 radially after it drops below 0.5%
        (the accuracy variable) of the peak density in
        the midplane, and vertically whenever it drops below 0.5% of the
        peak density in the midplane

        Parameters 
        -----------
        accuracy : float, optional
            cutoff for dust density distribution as a fraction of the peak density in the midplane (default 5e-3)
        ain : float, optional
            inner power law exponent. must be positive (default 5)
        aout : float, optional
            outer power law exponent. must be negative (default -5)
        a : float, optional
            disk reference radius in au as defined in VIP https://vip.readthedocs.io/en/latest/tutorials/05B_fm_disks.html (default 60)
        e : float, optional
            eccentricity, 0 < e < 1 (default 0)
        ksi0 : float, optional
            scale height in au at the reference radius (default 1)
        gamma : float, optional
            exponential decay exponent for vertical profile (2 = Gaussian, 1 = exponential profile, default 2)
        beta : float, optional
            flaring coefficient in scale height (0 = no flaring, 1 = linear flaring, default 1)
        amin : float, optional
            minimum semi-major axis; dust density is 0 below this value (default 0)
        dens_at_r0 : float, optional
            density at r0 (default 1)
        """

        p_dict = {}
        p_dict["accuracy"] = accuracy 

        p_dict["ksi0"] = ksi0
        p_dict["gamma"] = gamma
        p_dict["beta"] = beta
        p_dict["zmax"] = ksi0*(-jnp.log(p_dict["accuracy"]))**(1./(gamma+1e-8)) # maximum height z

        # Set Vertical Density Analogue
        gamma = jnp.where(gamma < 0., 0.1, gamma)
        ksi0 = jnp.where(ksi0 < 0., 0.1, ksi0)
        beta = jnp.where(beta < 0., 0., beta)

        # Set Radial Density Analogue
        ain = jnp.where(ain < 0.01, 0.01, ain)
        aout = jnp.where(aout > -0.01, -0.01, aout)
        e = jnp.where(e < 0., 0., e)
        e = jnp.where(e >= 1, 0.99, e)
        amin = jnp.where(amin < 0., 0., amin)
        dens_at_r0 = jnp.where(dens_at_r0 < 0., 0., dens_at_r0)

        p_dict["ain"] = ain
        p_dict["aout"] = aout
        p_dict["a"] = a
        p_dict["e"] = e
        p_dict["p"] = p_dict["a"]*(1-p_dict["e"]**2) # ellipse parameter p (see VIP documentation)
        p_dict["amin"] = amin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["amin"]*(1-p_dict["e"]**2) # WHAT IS THIS
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["a"]*p_dict["accuracy"]**(1/(p_dict["aout"]+1e-8)) # maximum radial distance
        p_dict["apeak"] = p_dict["a"] * jnp.power(-p_dict["ain"]/(p_dict["aout"]+1e-8),
                                        1./(2.*(p_dict["ain"]-p_dict["aout"]))) #peak radius
        Gamma_in = jnp.abs(p_dict["ain"]+p_dict["beta"] + 1e-8) # as defined in Augereau et al 1999
        Gamma_out = -jnp.abs(p_dict["aout"]+p_dict["beta"] + 1e-8) # as defined in Augereau et al 1999
        p_dict["apeak_surface_density"] = p_dict["a"] * jnp.power(-Gamma_in/Gamma_out,
                                                        1./(2.*(Gamma_in-Gamma_out+1e-8)))
        # the above formula comes from Augereau et al. 1999.
        p_dict["itiltthreshold"] = jnp.rad2deg(jnp.arctan(p_dict["rmax"]/p_dict["zmax"]))

        return cls.pack_pars(p_dict)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particle volume density at a given location in cylindrical coordinates (r, theta, z) in the disk 

        Parameters
        -----------
        distr_params : array
            numpy array of input model parameters
        r : float
            radial distance from the star
        costheta : float
            cosine of angle around the disk
        z : float
            vertical height from the midplane

        Returns
        -----------
        density_term : float
            particle volume density at r, theta, z
        """
        distr = cls.unpack_pars(distr_params)

        radial_ratio = r*(1-distr["e"]*costheta)/((distr["p"])+1e-8)

        den = (jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["ain"]) +
               jnp.power(jnp.abs(radial_ratio)+1e-8, -2*distr["aout"]))
        radial_density_term = jnp.sqrt(2./den+1e-8)*distr["dens_at_r0"]
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

    Methods
    ----------
    compute_phase_function_from_cosphi
    """

    params = {'g': 0.3}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        Constructor of a Heyney Greenstein phase function.

        Parameters
        ----------
        func_params :  dictionary containing the key "g" (float)
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
        phase_func_params : float or array
            input parameters as defined in the class constructor
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.

        Returns
        ----------
        HG phase function at the given scattering angles
        """
        p_dict = cls.unpack_pars(phase_func_params)
        
        return 1./(4*jnp.pi)*(1-p_dict["g"]**2) / \
            (1+p_dict["g"]**2-2*p_dict["g"]*cos_phi)**(3./2.)


class DoubleHenyeyGreenstein_SPF(Jax_class):
    """
    Implementation of a scattering phase function with a double Henyey
    Greenstein function.

    Methods
    ----------
    compute_phase_function_from_cosphi
    """

    params = {'g1': 0.5, 'g2': -0.3, 'weight': 0.7}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, func_params):
        """
        Constructor of a Double Heyney Greenstein phase function.

        Parameters
        ----------
        func_params :  dictionary containing the keys "g1" (float), "g2" (float), "weight" (float)
            g is the Heyney Greenstein coefficient and should be between -1
            (backward scattering) and 1 (forward scattering) for both phase functions, as well as the weighting factor

        weighting factor will multiply the first HG coefficient (g1) and (1 - weight) will multiply with g2
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
        phase_func_params : float or array
            input parameters as defined in the class constructor
        cos_phi : float or array
            cosine of the scattering angle(s) at which the scattering function
            must be calculated.

        Returns
        ----------
        Double HG phase function at the given scattering angles
        """

        p_dict = cls.unpack_pars(phase_func_params)

        hg1 = p_dict['weight'] * 1./(4*jnp.pi)*(1-p_dict["g1"]**2) / \
            (1+p_dict["g1"]**2-2*p_dict["g1"]*cos_phi)**(3./2.)
        hg2 = (1-p_dict['weight']) * 1./(4*jnp.pi)*(1-p_dict["g2"]**2) / \
            (1+p_dict["g2"]**2-2*p_dict["g2"]*cos_phi)**(3./2.)
        
        return hg1+hg2
    

# Uses 10 knots by default
# Values must be cos(phi) not phi
class InterpolatedUnivariateSpline_SPF(Jax_class):
    """
    Implementation of a spline scattering phase function. Uses 6 knots by default, takes knot y values as parameters.
    Locations are fixed to the given knots, pack_pars and init both return the spline model itself

    Methods
    ----------
    unpack_pars
    pack_pars
    compute_phase_function_from_cosphi
    """

    params = jnp.ones(6)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """Helper function to unpack parameters -- replaces Jax_class definition of this function to be specific for spline SPF"""
        return p_arr

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, p_arr, knots=jnp.linspace(1, -1, 6)):
        """
        This function takes a array of (knots) values and converts them into an InterpolatedUnivariateSpline model.
        Also has inclination bounds which help narrow the spline fit. 
        
        Note: appears to duplicate functionality of init below, but this is on purpose!
        It's because this particular class is implemented through pytrees from jax scipy, 
        so it's just wrapped to make it compatible with the structure of the other SPF functions

        Parameters
        ----------
        p_arr : array
            input parameters as defined in the class constructor; y values for knots and should be same length array as knots
        knots : jax array
            x values for knots; nominally a jax array of format jnp.linspace(1,-1,n) where n is the desired number of knots

        Returns
        ----------
        Interpolated spline for the given knots
        """    
        
        y_vals = p_arr
        return InterpolatedUnivariateSpline(knots, y_vals)

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def init(cls, p_arr, knots=jnp.linspace(1, -1, 6)):
        """ Class constructor for the Interpolated Univariate Spline

        Parameters
        ----------
        p_arr : array
            input parameters as defined in the class constructor; y values for knots and should be same length array as knots
        knots : jax array
            x values for knots; nominally a jax array of format jnp.linspace(1,-1,n) where n is the desired number of knots

        Returns
        ----------
        Interpolated spline for the given knots
        """

        y_vals = p_arr
        return InterpolatedUnivariateSpline(knots, y_vals)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
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

        Returns
        ----------
        Spline model phase function at a given angle
        """
        
        return spline_model(cos_phi)
    

class GAUSSIAN_PSF(Jax_class):
    """ Generates a Gaussian PSF for convolution

    Methods
    ----------
    generate
    """
    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3,4,5,6,7))
    def generate(cls, pos, FWHM = 3, xo = 0., yo = 0., theta=0, offset=0, amplitude=1):
        """
        Function to make a Gaussian PSF with a given FWHM and other parameters

        Parameters
        ----------
        pos : numpy meshgrid
            x and y coordinates over which the psf should be evaluated
        FWHM : float, optional
            full width at half maximum (default 3)
        xo : float, optional
            x offset shift, redundant with "offset" (default 0)
        yo : float, optional
            y offset shift, redundant with "offset" (default 0)
        theta : float, optional
            angle to rotate output PSF; for axisymmetric, theta = 0
        offset : float, optional
            xy offset, redundant with xo and yo
        amplitude : float, optional
            amplitude of the gaussian PSF (default 1)

        Returns
        ----------
        Gaussian PSF, array
        
        """
        sigma = FWHM / 2.355
        a = (jnp.cos(theta)**2)/(2*sigma**2) + (jnp.sin(theta)**2)/(2*sigma**2)
        b = -(jnp.sin(2*theta))/(4*sigma**2) + (jnp.sin(2*theta))/(4*sigma**2)
        c = (jnp.sin(theta)**2)/(2*sigma**2) + (jnp.cos(theta)**2)/(2*sigma**2)
        return offset + amplitude*jnp.exp( - (a*((pos[0]-xo)**2) + 2*b*(pos[0]-xo)
                                                                      *(pos[1]-yo) + c*((pos[1]-yo)**2)))
    

class EMP_PSF(Jax_class):
    """ Generates an empirical PSF from an image

    Methods
    ----------
    process_image
    generate
    """
    def process_image(image, scale_factor=1, offset=1):
        """
        Processes an input image to create an empirical PSF

        Parameters
        ----------
        image : array-like
            the input image from which to generate the empirical PSF; note: currently crops image with hardcoded values
        scale_factor : float, optional
            factor by which to scale the input image
        offset : float, optional
            how much to offset the image (value doesn't actually do anything right now, not implemented)

        Returns
        ----------
        PSF image, array
        """
        scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
        cropped_image = image[70:210, 70:210]
        def safe_float32_conversion(value):
            try:
                return np.float32(value)
            except (ValueError, TypeError):
                print("This value is unjaxable: " + str(value))
        fin_image = np.nan_to_num(cropped_image)
        fin_image = np.vectorize(safe_float32_conversion)(fin_image)
        return fin_image

    img = process_image(fits.open("PSF/emp_psf.fits")[0].data[0,:,:])

    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, pos):
        """
        Generates empirical PSF

        Parameters
        ----------
        pos : numpy meshgrid
            x and y coordinates over which the psf should be evaluated

        Returns
        ----------
        PSF image, array
        
        """
        return cls.img