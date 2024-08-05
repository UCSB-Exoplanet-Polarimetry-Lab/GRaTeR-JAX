import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from interpolated_univariate_spline import InterpolatedUnivariateSpline
from astropy.io import fits
from pyklip import klip

class Jax_class:

    params = {}

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        This function takes a parameter array (params) and unpacks it into a
        dictionary with the parameter names as keys.
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
        """    
        p_arrs = []
        for name in cls.params.keys():
            p_arrs.append(p_dict[name])
        return jnp.asarray(p_arrs)


class DustEllipticalDistribution2PowerLaws(Jax_class):
    """
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
        ain = jnp.where(ain < 0.1, 0.1, ain)
        aout = jnp.where(aout > -0.1, -0.1, aout)
        e = jnp.where(e < 0., 0., e)
        e = jnp.where(e >= 1, 0.99, e)
        amin = jnp.where(amin < 0., 0., amin)
        dens_at_r0 = jnp.where(dens_at_r0 < 0., 1., dens_at_r0)

        p_dict["ain"] = ain
        p_dict["aout"] = aout
        p_dict["a"] = a
        p_dict["e"] = e
        p_dict["p"] = p_dict["a"]*(1-p_dict["e"]**2)
        p_dict["amin"] = amin
        # we assume the inner hole is also elliptic (convention)
        p_dict["pmin"] = p_dict["amin"]*(1-p_dict["e"]**2)
        p_dict["dens_at_r0"] = dens_at_r0

        # maximum distance of integration, AU
        p_dict["rmax"] = p_dict["a"]*p_dict["accuracy"]**(1/(p_dict["aout"]+1e-8))
        p_dict["apeak"] = p_dict["a"] * jnp.power(-p_dict["ain"]/(p_dict["aout"]+1e-8),
                                        1./(2.*(p_dict["ain"]-p_dict["aout"])))
        Gamma_in = jnp.abs(p_dict["ain"]+p_dict["beta"] + 1e-8)
        Gamma_out = -jnp.abs(p_dict["aout"]+p_dict["beta"] + 1e-8)
        p_dict["apeak_surface_density"] = p_dict["a"] * jnp.power(-Gamma_in/Gamma_out,
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
    
    '''@classmethod
    @partial(jax.jit, static_argnums=(0,))
    def density_cylindrical(cls, distr_params, r, costheta, z):
        """ Returns the particle volume density at r, theta, z """
        distr = cls.unpack_pars(distr_params)

        radial_ratio = r * (1 - distr["e"] * costheta) / (distr["p"] + 1e-8)
        jax.debug.print("radial_ratio: {}", jnp.sum(jnp.where(jnp.isnan(radial_ratio), 1, 0)))

        den = (jnp.power(jnp.abs(radial_ratio) + 1e-8, -2 * distr["ain"] + 1e-8) +
            jnp.power(jnp.abs(radial_ratio) + 1e-8, -2 * distr["aout"] + 1e-8))
        jax.debug.print("den: {}", jnp.sum(jnp.where(jnp.isnan(den), 1, 0)))
        
        radial_density_term = jnp.sqrt(2. / den + 1e-8) * distr["dens_at_r0"]
        jax.debug.print("radial_density_term (before): {}", jnp.sum(jnp.where(jnp.isnan(radial_density_term), 1, 0)))

        radial_density_term = jnp.where(distr["pmin"] > 0, 
                                        jnp.where(r * (1 - distr["e"] * costheta) / (distr["p"] + 1e-8) <= 1, 0., radial_density_term),
                                        radial_density_term)
        jax.debug.print("radial_density_term (after): {}", jnp.sum(jnp.where(jnp.isnan(radial_density_term), 1, 0)))

        den2 = distr["ksi0"] * jnp.power(jnp.abs(radial_ratio + 1e-8), distr["beta"]) + 1e-8
        jax.debug.print("den2: {}", den2)

        vertical_density_term = jnp.exp(-jnp.power((jnp.abs(z) + 1e-8) / (jnp.abs(den2 + 1e-8)), jnp.abs(distr["gamma"]) + 1e-8))
        jax.debug.print("vertical_density_term: {}", jnp.sum(jnp.where(jnp.isnan(vertical_density_term), 1, 0)))
        
        result = radial_density_term * vertical_density_term
        jax.debug.print("result: {}", jnp.sum(jnp.where(jnp.isnan(result), 1, 0)))
        
        return result'''

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
        spf_dico :  dictionnary containing the key "g" (float)
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
    

# Uses 10 knots by default
# Values must be cos(phi) not phi
class InterpolatedUnivariateSpline_SPF(Jax_class):
    """
    Implementation of a spline scattering phase function. Uses 6 knots by default, takes knot y values as parameters.
    Locations are fixed to the given knots, pack_pars and init both return the spline model itself
    """

    params = jnp.ones(6)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        return p_arr

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def pack_pars(cls, p_arr, knots=jnp.linspace(1, -1, 6)):
        """
        This function takes a array of (knots) values and converts them into an InterpolatedUnivariateSpline model.
        Also has inclination bounds which help narrow the spline fit
        """    
        
        y_vals = p_arr
        return InterpolatedUnivariateSpline(knots, y_vals)

    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def init(cls, p_arr, knots=jnp.linspace(1, -1, 6)):
        """
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
        """
        
        return spline_model(cos_phi)
    

class GAUSSIAN_PSF(Jax_class):

    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0,2,3,4,5,6,7))
    def generate(cls, pos, FWHM = 3, xo = 0., yo = 0., theta=0, offset=0, amplitude=1):
        sigma = FWHM / 2.355
        a = (jnp.cos(theta)**2)/(2*sigma**2) + (jnp.sin(theta)**2)/(2*sigma**2)
        b = -(jnp.sin(2*theta))/(4*sigma**2) + (jnp.sin(2*theta))/(4*sigma**2)
        c = (jnp.sin(theta)**2)/(2*sigma**2) + (jnp.cos(theta)**2)/(2*sigma**2)
        return offset + amplitude*jnp.exp( - (a*((pos[0]-xo)**2) + 2*b*(pos[0]-xo)
                                                                      *(pos[1]-yo) + c*((pos[1]-yo)**2)))
    

class EMP_PSF(Jax_class):

    def process_image(image, scale_factor=1, offset=1):
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
    
    psf_hdu = fits.open("PSF/emp_psf.fits")
    psf_data = psf_hdu[0].data.astype(jnp.float32)

    img = process_image(jnp.median(psf_data,axis=0))
    
    #define model function and pass independant variables x and y as a list
    @classmethod
    @partial(jax.jit, static_argnums=(0))
    def generate(cls, pos):
        return cls.img