"""
sld_class.py
============

Scattered-light disk model class.

This module defines the `ScatteredLightDisk` class, a JAX-accelerated
implementation of the GRaTeR framework for generating synthetic scattered-light
disk images. It provides utilities for initializing disk parameters,
packing/unpacking arrays for JAX, and computing forward-modeled scattered-light
images for debris and protoplanetary disks.
"""

import jax
import jax.numpy as jnp
from functools import partial
from grater_jax.disk_model.SLD_utils import *

class ScatteredLightDisk(Jax_class):
    """
    Generate a synthetic scattered-light disk using a lightweight, JAX-accelerated
    version of the GRaTeR approach.

    This class provides utilities to construct surface-brightness images of inclined,
    possibly eccentric debris/protoplanetary disks for forward modeling and fitting.

    Parameters
    ----------
    nx : int, optional
        Number of pixels along the x axis (default 200).
    ny : int, optional
        Number of pixels along the y axis (default 200).
    distance : float, optional
        Distance to the star in parsecs (default 70.0).
    itilt : float
        Inclination with respect to the line of sight in degrees
        (0 = pole-on; 90 = edge-on).
    omega : float, optional
        Argument of pericenter in degrees (default 0).
    pxInArcsec : float, optional
        Pixel scale in arcsec/pixel (default 0.01225).
    pa : float
        Disk position angle in degrees.
        PA is measured from North (positive y axis) increasing counter-clockwise
        toward the projected major axis of the disk. With this convention,
        ``PA - 90°`` equals the PA of the projected semi-minor axis on the
        presumed *front* (brighter) side. This is offset by 180° from
        Esposito et al. (2020) and matches the convention used in VIP’s
        forward-modeling utilities.
    distr_params : dict
        Dictionary controlling the dust density distribution. See
        **“distr_params keys”** below for expected fields.
    xdo : float, optional
        Disk offset along the x axis of the disk frame (AU; default 0).
    ydo : float, optional
        Disk offset along the y axis of the disk frame (AU; default 0).

    distr_params keys
    -----------------
    accuracy : float, optional
        Density cutoff used for truncation (default ``5e-3``).
    alpha_in : float
        Power-law slope of the *inner* disk (must be positive; default 5).
    alpha_out : float
        Power-law slope of the *outer* disk (must be negative; default -5).
    sma : float
        Reference radius in AU (default 60).
    e : float, optional
        Eccentricity (default 0).
    ksi0 : float, optional
        Scale height at the reference radius in AU (default 1).
    gamma : float, optional
        Vertical profile exponent (2 = Gaussian, 1 = exponential; default 2).
    beta : float, optional
        Flaring index (0 = none, 1 = linear; default 1).
    rmin : float, optional
        Minimum semi-major axis; density is zero below this value (default 0).

    Notes
    -----
    The class stores its parameters in a packed array for JAX and provides
    helpers to pack/unpack dictionaries.
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
    def init(cls, distr_params, itilt, pa, alpha_in, alpha_out, sma, nx=200, ny=200, distance=50., omega=0., pxInArcsec=0.01225, xdo=0., ydo=0.):
        """Compute derived geometric parameters and pack them into a distribution parameter array.

        Parameters
        ----------
        distr_params : jax.numpy.ndarray
            Existing packed distribution parameter array (used to resolve inclination ambiguity).
        itilt : float
            Disk inclination with respect to the line of sight, in degrees.
        pa : float
            Position angle of the disk, in degrees.
        alpha_in : float
            Inner power-law index of the dust density profile.
        alpha_out : float
            Outer power-law index of the dust density profile.
        sma : float
            Semi-major axis (ring radius) of the disk, in AU.
        nx : int, optional
            Number of pixels along the x axis. Default is 200.
        ny : int, optional
            Number of pixels along the y axis. Default is 200.
        distance : float, optional
            Distance to the star in parsecs. Default is 50.
        omega : float, optional
            Argument of periapsis, in degrees. Default is 0.
        pxInArcsec : float, optional
            Pixel scale in arcsec/pixel. Default is 0.01225.
        xdo : float, optional
            Disk offset along the semi-major axis, in AU. Default is 0.
        ydo : float, optional
            Disk offset along the semi-minor axis, in AU. Default is 0.

        Returns
        -------
        jax.numpy.ndarray
            Packed distribution parameter array.
        """
        p_dict = {}

        p_dict["nx"] = nx    # number of pixels along the x axis of the image
        p_dict["ny"] = ny    # number of pixels along the y axis of the image
        p_dict["distance"] = distance  # distance to the star in pc
        p_dict["omega"] = omega

        p_dict["pxInArcsec"] = pxInArcsec  # pixel field of view in arcsec/px
        p_dict["pxInAU"] = p_dict["pxInArcsec"]*p_dict["distance"]     # 1 pixel in AU
        # disk offset along the x-axis in the disk frame (semi-major axis), AU
        p_dict["xdo"] = xdo
        # disk offset along the y-axis in the disk frame (semi-minor axis), AU
        p_dict["ydo"] = ydo
        p_dict["rmin"] = jnp.sqrt(p_dict["xdo"]**2+p_dict["ydo"]**2)+p_dict["pxInAU"]
        # star center along the y- and x-axis, in pixels

        p_dict["itilt"] = itilt  # inclination wrt the line of sight in deg
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
            Parameters describing the disk, see above for full description
        distr_params : dict
            Parameters describing the dust density distribution function, also defined above
        distr_cls : class
            Class describing the dust density distribution function (nominally should be DustEllipticalDistribution2PowerLaws,
            other distributions are not yet implemented)
        phase_func_params : dict
            Parameters describing the phase function, see your respective phase function class for full description
        phase_func_cls : class
            Class describing the phase function (options are HenyeyGreenstein_SPF, DoubleHenyeyGreenstein_SPF, InterpolatedUnivariateSpline_SPF)
        halfNbSlices : integer
            half number of distances along the line of sight
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
        validPixel_map = (lmax2 > 0.) * (lmin2 > 1e-6 * disk["rmin"]**2)
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

        # Vectorized line-of-sight integration: compute all slices at once.
        # ll: (nbSlices,), maps: (ny, nx) → intermediates: (nbSlices, ny, nx)
        # Distance along the line of sight to each plane z
        l_all = jnp.where(validPixel_map,
                          lz0_map + ll[:, None, None] * dl_map, 0)

        # rotation about x axis
        yd_all = ycs_vector + disk["sini"] * l_all   # y_Disk in AU
        zd_all = zsn_vector + disk["cosi"] * l_all   # z_Disk in AU

        # Dist and polar angles in the frame centered on the star position
        d2star_all = xd_vector**2 + yd_all**2 + zd_all**2
        dstar_all = jnp.sqrt(d2star_all + 1e-8)
        rstar_all = jnp.sqrt(xd_vector**2 + yd_all**2 + 1e-8)
        thetastar_all = jnp.arctan2(yd_all, xd_vector + 1e-8)

        # Phase angles
        cosphi_all = (rstar_all * disk["sini"] * jnp.sin(thetastar_all) +
                      zd_all * disk["cosi"]) / (dstar_all + 1e-8)

        # Polar coordinates in the disk frame
        r_all = jnp.sqrt((xd_vector - disk["xdo"])**2 +
                         (yd_all - disk["ydo"])**2 + 1e-8)
        theta_all = jnp.arctan2(yd_all - disk["ydo"],
                                xd_vector - disk["xdo"] + 1e-8)
        costheta_all = jnp.cos(theta_all - jnp.deg2rad(disk["omega"]))

        # Scattered light: density × phase function / distance²
        rho_all = distr_cls.density_cylindrical(
            distr_params, r_all, costheta_all, zd_all
        )
        phase_all = phase_func_cls.compute_phase_function_from_cosphi(
            phase_func_params, cosphi_all
        )
        limage = jnp.where(validPixel_map,
                           rho_all * phase_all / (d2star_all + 1e-8), 0)

        # Trapezoidal integration along line of sight
        dl = ll[1:] - ll[:-1]                      # (nbSlices-1,)
        pair_sum = limage[:-1] + limage[1:]        # (nbSlices-1, ny, nx)

        scattered_light_map = jnp.sum(dl[:, None, None] * pair_sum, axis=0)
        scattered_light_map = jnp.where(validPixel_map, scattered_light_map * dl_map / 2. * disk["pxInAU"]**2, 0)

        #ideally should check for valid pixel map
        #if disk["flux_max"] is not None:
        #    scattered_light_map = scattered_light_map * (disk["flux_max"] /
        #                                 jnp.nanmax(scattered_light_map))
            
        return scattered_light_map