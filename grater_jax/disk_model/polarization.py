"""Single-scattering linear polarization in image coordinates.

Angles increase from +x toward +y (increasing array row). Q is positive for
polarization along x; U is positive at +45 degrees. Qphi is positive for
azimuthal polarization. These are array-coordinate conventions; callers must
account for sky orientation when comparing to reduced observations.
"""

import jax.numpy as jnp


def azimuthal_factors(shape, star_x, star_y):
    """Return cos(2 phi), sin(2 phi); set both to zero at the star.

    Rational expressions avoid the undefined derivative of atan2 at the origin.
    """
    y, x = jnp.indices(shape, dtype=jnp.result_type(star_x, star_y, 1.0))
    dx, dy = x - star_x, y - star_y
    r2 = dx * dx + dy * dy
    denominator = jnp.where(r2 > 0, r2, 1.0)
    return (dx * dx - dy * dy) / denominator, 2 * dx * dy / denominator


def polarized_intensity_to_stokes(image, star_x, star_y):
    """Convert intrinsic polarized intensity into tangential Q and U.

    The stellar pixel is assigned zero polarization. The input must model
    polarized intensity, rather than total intensity.
    """
    c, s = azimuthal_factors(image.shape, star_x, star_y)
    return -image * c, -image * s


def stokes_to_azimuthal(q, u, star_x, star_y):
    """Transform Q,U into Qphi,Uphi using a tangential-positive convention."""
    c, s = azimuthal_factors(q.shape, star_x, star_y)
    return -q * c - u * s, q * s - u * c


def polarized_image(image, star_x, star_y, psf_model, psf_params,
                    flux_scaling, throughput, stellar_model, output):
    """Build one observable polarization product from the intrinsic disk."""
    if output not in ('Q', 'U', 'Qphi', 'Uphi', 'PI'):
        raise ValueError("output must be 'image', 'PI', 'Q', 'U', 'Qphi', or 'Uphi'")
    if stellar_model is not None:
        raise ValueError("Polarization outputs require StellarPSFModel=None; stellar intensity is not a Stokes model")
    q, u = polarized_intensity_to_stokes(image, star_x, star_y)
    if psf_model is not None:
        # Remove any additive PSF background: only the linear response belongs
        # in the convolution of signed Stokes signals (e.g. GAUSSIAN_PSF.offset).
        background = psf_model.generate(jnp.zeros_like(image), psf_params)
        q = psf_model.generate(q, psf_params) - background
        u = psf_model.generate(u, psf_params) - background
    q, u = q * flux_scaling * throughput, u * flux_scaling * throughput
    if output == 'Q':
        return q
    if output == 'U':
        return u
    if output == 'PI':
        power = q * q + u * u
        # Define a finite zero subgradient at pixels with no polarization.
        return jnp.where(power > 0, jnp.sqrt(jnp.where(power > 0, power, 1.0)), 0.0)
    qphi, uphi = stokes_to_azimuthal(q, u, star_x, star_y)
    return qphi if output == 'Qphi' else uphi
