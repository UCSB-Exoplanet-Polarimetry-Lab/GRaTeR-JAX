#! /usr/bin/env python
"""
Class definition for ScatteredLightDisk -- Dust_distribution and Phase_function are in SLD_utils

.. [AUG99]
   | Augereau et al. 1999
   | **On the HR 4796 A circumstellar disk**
   | *Astronomy & Astrophysics, Volume 348, pp. 557-569*
   | `https://arxiv.org/abs/astro-ph/9906429
     <https://arxiv.org/abs/astro-ph/9906429>`_

"""

__author__ = 'Julien Milli'
__all__ = ['ScatteredLightDisk',
           'Dust_distribution',
           'Phase_function']

import jax
import jax.numpy as jnp
from functools import partial
from SLD_utils import *


class ScatteredLightDisk(Jax_class):
    """
    Class used to generate a synthetic disc, inspired from a light version of
    the GRATER tool (GRenoble RAdiative TransfER) written originally in IDL
    [AUG99]_, and converted to Python by J. Milli.
    """

    # Jax Parameters
    params = {
        "nx": 0, "ny": 0,
        "distance": 0.,
        "itilt": 0.,
        "omega": 0.,
        "pxInArcsec": 0.,
        "pxInAU": 0.,
        "pa": 0.,
        "xdo": 0., "ydo": 0.,
        "rmin": 0.,
        "cospa": 0., "sinpa": 0.,
        "cosi": 0., "sini": 0.
    }

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def init(cls, distr_params, inc, pa, ain, aout, sma, nx=200, ny=200, distance=50., omega=0., pxInArcsec=0.01225, xdo=0., ydo=0.):
        """Initializes ScatteredLightDisk with given parameters
        
        Parameters
        ----------
        distr_params : dict
            Parameters describing the dust density distribution function
            to be implemented. Currently, only a two power law distribution is implemented 
            with the following parameters:
                a : float
                    reference radius in au (default 40)
                ksi0 : float
                    scale height in au at the reference radius (default 1 a.u.)
                gamma : float
                    exponent (2=gaussian,1=exponential profile, default 2)
                beta : float
                    flaring index (0=no flaring, 1=linear flaring, default 1)
                ain : float
                    slope of the power-low distribution in the inner disk. It
                    must be positive (default 5)
                aout : float
                    slope of the power-low distribution in the outer disk. It
                    must be negative (default -5)
                e : float
                    eccentricity (default 0)
                amin: float
                    minimim semi-major axis: the dust density is 0 below this
                    value (default 0)
        nx : int
            number of pixels along the x axis of the image (default 200)
        ny : int
            number of pixels along the y axis of the image (default 200)
        distance : float
            distance to the star in pc (default 70.)
        inc : float
            inclination wrt the line of sight in degrees (0 means pole-on,
            90 means edge-on, default 60 degrees)
        omega : float
            argument of the pericenter in degrees (0 by default)
        pxInArcsec : float
            pixel field of view in arcsec/px (default the SPHERE pixel
            scale 0.01225 arcsec/px)
        pa : float
            position angle of the disc in degrees (default 0 degrees, e.g. North)
        xdo : float
            disk offset along the x-axis in the disk frame (=semi-major axis),
            in a.u. (default 0)
        ydo : float
            disk offset along the y-axis in the disk frame (=semi-minor axis),
            in a.u. (default 0)
            """
        p_dict = {}

        p_dict["nx"] = nx    # number of pixels along the x axis of the image
        p_dict["ny"] = ny    # number of pixels along the y axis of the image
        p_dict["distance"] = distance  # distance to the star in pc
        p_dict["omega"] = omega

        p_dict["pxInArcsec"] = pxInArcsec  # pixel field of view in arcsec/px
        p_dict["pxInAU"] = p_dict["pxInArcsec"]*p_dict["distance"] # 1 pixel in AU
        p_dict["xdo"] = xdo # disk offset along the x-axis in the disk frame (semi-major axis), AU
        p_dict["ydo"] = ydo # disk offset along the y-axis in the disk frame (semi-minor axis), AU
        p_dict["rmin"] = jnp.sqrt(p_dict["xdo"]**2+p_dict["ydo"]**2)+p_dict["pxInAU"] # star center along the y- and x-axis, in pixels

        p_dict["itilt"] = inc  # inclination wrt the line of sight in deg
        p_dict["cosi"] = jnp.cos(jnp.deg2rad(p_dict["itilt"]))
        p_dict["sini"] = jnp.sin(jnp.deg2rad(p_dict["itilt"]))

        p_dict["pa"] = pa    # position angle of the disc in degrees
        p_dict["cospa"] = jnp.cos(jnp.deg2rad(pa))
        p_dict["sinpa"] = jnp.sin(jnp.deg2rad(pa))

        p_dict["itilt"] = jnp.where(jnp.abs(jnp.mod(p_dict["itilt"], 180)-90) < jnp.abs(
                jnp.mod(distr_params[16], 180)-90),
                distr_params[16], p_dict["itilt"]) 
        
        return cls.pack_pars(p_dict)
        

    @classmethod
    @partial(jax.jit, static_argnums=(0, 3, 5, 12))
    def compute_scattered_light_jax(cls, disk_params, distr_params, distr_cls, phase_func_params,
                                    phase_func_cls, x_vector, y_vector, scattered_light_map,
                                    image, limage, tmp, halfNbSlices):
        """
        Computes the scattered light image of the disk.

        Parameters
        ----------
        disk_params : dict
            dictionary of ScatteredLightDisk model parameters
        distr_params : dict
            dictionary of dust distribution parameters as follows:
                a : float
                    reference radius in au (default 40)
                ksi0 : float
                    scale height in au at the reference radius (default 1 a.u.)
                gamma : float
                    exponent (2=gaussian,1=exponential profile, default 2)
                beta : float
                    flaring index (0=no flaring, 1=linear flaring, default 1)
                ain : float
                    slope of the power-low distribution in the inner disk. It
                    must be positive (default 5)
                aout : float
                    slope of the power-low distribution in the outer disk. It
                    must be negative (default -5)
                e : float
                    eccentricity (default 0)
                amin: float
                    minimim semi-major axis: the dust density is 0 below this
                    value (default 0)
        distr_cls : DustEllipticalDistribution2PowerLaws
            dust distribution class; only two power law distribution is implemented
        phase_func_params : dict
            phase function parameters for your chosen type of SPF (g if single HG, g1/g2/weight if double HG, knots if spline)
        phase_func_cls : class
            your choice of phase function class (HG, Double HG, Spline) from SLD_utils
        x_vector : array
            x_disk in AU
        y_vector : array
            y_disk in AU
        scattered_light_map : jax array
            array of zeros with size nx, ny; exists because jax requires traced values
        image : jax array
            array of zeros with size nx, ny; exists because jax requires traced values
        limage : jax array
            array of zeros with size 2*halfNbSlices-1, nx, ny; exists because jax requires traced values
        tmp : jax array
            array of values 0 to halfNbSlices; exists because jax requires traced values
        halfNbSlices : integer
            half number of distances along the line of sight l

        Returns
        ----------
        scattered_light_map : array
            model disk image
        """

        disk = cls.unpack_pars(disk_params)

        distr = distr_cls.unpack_pars(distr_params)

        #x_vector = (jnp.arange(0, disk["nx"]) - disk["xc"])*disk["pxInAU"]  # x axis in au
        #y_vector = (jnp.arange(0, disk["ny"]) - disk["yc"])*disk["pxInAU"]  # y axis in au
        
        x_map_0PA, y_map_0PA = jnp.meshgrid(x_vector, y_vector)
        # rotation to get the disk major axis properly oriented, x in AU
        y_map = (disk["cospa"]*x_map_0PA + disk["sinpa"]*y_map_0PA)
        # rotation to get the disk major axis properly oriented, y in AU
        x_map = (-disk["sinpa"]*x_map_0PA + disk["cospa"]*y_map_0PA)

        # dist along the line of sight to reach the disk midplane (z_D=0), AU:
        lz0_map = y_map * jnp.tan(jnp.deg2rad(disk["itilt"]))
        # dist to reach +zmax, AU:
        lzp_map = distr["zmax"]/disk["cosi"] + \
            lz0_map
        # dist to reach -zmax, AU:
        lzm_map = -distr["zmax"]/disk["cosi"] + \
            lz0_map
        dl_map = jnp.absolute(lzp_map-lzm_map)  # l range, in AU
        # squared maximum l value to reach the outer disk radius, in AU^2:
        lmax2 = distr["rmax"]**2 - \
            (x_map**2+y_map**2)
        # squared minimum l value to reach the inner disk radius, in AU^2:
        lmin2 = (x_map**2+y_map**2)-disk["rmin"]**2
        validPixel_map = (lmax2 > 0.) * (lmin2 > 0.)
        lwidth = 100.  # control the distribution of distances along l
        nbSlices = 2*halfNbSlices-1
        # total number of distances
        # along the line of sight

        tmp = (jnp.exp(tmp*jnp.log(lwidth+1.) /
                      (halfNbSlices-1.))-1.)/lwidth  # between 0 and 1
        
        ll = jnp.concatenate((-tmp[:0:-1], tmp))

        # 1d array pre-calculated values, AU
        ycs_vector = jnp.where(validPixel_map, disk["cosi"]*y_map, 0)
        # 1d array pre-calculated values, AU
        zsn_vector = jnp.where(validPixel_map, -disk["sini"]*y_map, 0)
        xd_vector = jnp.where(validPixel_map, x_map, 0)  # x_disk, in AU

        #limage = jnp.zeros([nbSlices, ny, nx])
        #image = jnp.zeros((ny, nx))

        for il in range(nbSlices):
            # distance along the line of sight to reach the plane z

            l_vector = jnp.where(validPixel_map, lz0_map + ll[il]*dl_map, 0)
            #l_vector = lz0_map + ll[il]*dl_map

            # rotation about x axis
            yd_vector = ycs_vector + disk["sini"] * l_vector  # y_Disk in AU
            zd_vector = zsn_vector + disk["cosi"] * l_vector  # z_Disk, in AU
            # Dist and polar angles in the frame centered on the star position:
            # squared distance to the star, in AU^2
            d2star_vector = xd_vector**2+yd_vector**2+zd_vector**2
            dstar_vector = jnp.sqrt(d2star_vector + 1e-8)  # distance to the star, in AU
            # midplane distance to the star (r coordinate), in AU
            rstar_vector = jnp.sqrt(xd_vector**2+yd_vector**2+1e-8)
            thetastar_vector = jnp.arctan2(yd_vector, xd_vector+1e-8)
            # Phase angles:
            cosphi_vector = (rstar_vector*disk["sini"]*jnp.sin(thetastar_vector) +
                             zd_vector*disk["cosi"])/(dstar_vector+1e-8)  # in radians
            # Polar coordinates in the disk frame, and semi-major axis:
            # midplane distance to the disk center (r coordinate), in AU
            r_vector = jnp.sqrt((xd_vector-disk["xdo"])**2+(yd_vector-disk["ydo"])**2+1e-8)
            # polar angle in radians between 0 and pi
            theta_vector = jnp.arctan2(yd_vector-disk["ydo"], xd_vector-disk["xdo"]+1e-8)

            costheta_vector = jnp.cos(theta_vector-jnp.deg2rad(disk["omega"]))
            # Scattered light:
            # volume density
            rho_vector = distr_cls.density_cylindrical(distr_params, r_vector,
                                                               costheta_vector,
                                                               zd_vector)
            phase_function = phase_func_cls.compute_phase_function_from_cosphi(phase_func_params, cosphi_vector)
            #image = np.ndarray((disk["ny"], disk["nx"]))
            image = jnp.where(validPixel_map, rho_vector*phase_function/(d2star_vector + 1e-8), 0)
            #limage[il, :, :] = image
            limage = limage.at[il,:,:].set(image)

        for il in range(1, nbSlices):
            scattered_light_map += (ll[il]-ll[il-1]) * (limage[il-1, :, :] +
                                                             limage[il, :, :])
        scattered_light_map = jnp.where(validPixel_map, scattered_light_map * dl_map / 2. * disk["pxInAU"]**2, 0)

        #ideally should check for valid pixel map
        #if disk["flux_max"] is not None:
        #    scattered_light_map = scattered_light_map * (disk["flux_max"] /
        #                                 jnp.nanmax(scattered_light_map))
            
        return scattered_light_map