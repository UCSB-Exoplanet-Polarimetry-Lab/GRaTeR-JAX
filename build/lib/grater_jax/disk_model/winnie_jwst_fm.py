"""
winnie_jwst_fm.py
=================

JWST PSF grid and convolution utilities.

This module provides functions adapted from Kellen Lawson’s Winnie package for
generating and applying JWST NIRCam coronagraphic PSFs. It includes routines
for creating spatially varying PSF grids, convolving model images with PSFs,
handling coronagraph masks, and performing image rotations/padding on CPU or
GPU. GPU acceleration via CuPy is supported where available. This file is
only used for its generate_nircam_grid function. This could be deprecated as
more integrated support for Winnie could be developed.

Main functions
--------------
- `generate_nircam_psf_grid` : Build a grid of NIRCam PSFs and index maps.
- `convolve_with_spatial_psfs` : Apply spatially varying PSF convolution.
- `rotate_hypercube`, `pad_and_rotate_hypercube` : Rotate image cubes.
- `generate_jwst_mask_image` : Generate coronagraph mask transmission maps.
- GPU-enabled helpers (`rotate_image_gpu`, `psf_convolve_gpu`, etc.).
"""

import functools

import numpy as np
from scipy import ndimage, signal

try:
    import webbpsf_ext
    from webbpsf_ext import image_manip, coords
    from astropy.io import fits
    from joblib import Parallel, delayed
except ImportError:
    webbpsf_ext = None

"""
This file provides utilities derived from Kellen Lawson's Winnie package.
It is primarily used for `generate_nircam_psf_grid`.
"""

def raw_model_to_convolved_model_cube(
    input_im,
    parangs,
    psfs,
    psf_inds,
    im_mask=None,
    cent=None,
    use_gpu=False,
    ncores=-2,
):
    """
    Create a PSF-convolved sequence of model images at given parallactic angles.

    Parameters
    ----------
    input_im : ndarray, shape (ny, nx)
        Raw model image (oversampled if the PSFs are oversampled).
    parangs : ndarray, shape (n_T,)
        Parallactic angles in degrees. Each output slice is rotated by the
        corresponding angle.
    psfs : ndarray, shape (n_psf, ny, nx)
        Grid of spatially sampled PSFs used *after* rotation. If these PSFs are
        oversampled, ``input_im`` should be oversampled by the same factor.
    psf_inds : ndarray, shape (ny, nx) or (n_T, ny, nx)
        If 2-D: index map selecting the PSF slice in ``psfs`` for each pixel.
        If 3-D: a different index map for each roll (matching ``parangs``).
    im_mask : ndarray, optional
        Coronagraph transmission map(s) multiplied into ``input_im`` before
        convolution. Shape as ``input_im`` (or stacked per roll if 3-D).
    cent : array_like, optional
        Cartesian pixel coordinate ``[x, y]`` of the star in ``input_im``.
    use_gpu : bool, default False
        Use GPU (CuPy) for rotation/convolution where applicable.
    ncores : int, default -2
        CPU threads for rotation when ``use_gpu=False``; ``-2`` uses all but
        one available core.

    Returns
    -------
    imcube : ndarray, shape (n_T, ny, nx)
        For each angle in ``parangs``, the rotated model convolved with the
        appropriate PSF.
    """
    nT = len(parangs)
    ny, nx = input_im.shape[-2:]
    uni_angs = np.unique(parangs)

    inp_rot_uni = rotate_hypercube(
        np.tile(input_im, (len(uni_angs), 1, 1)),
        uni_angs,
        cent=cent,
        ncores=ncores,
        use_gpu=use_gpu,
        cval0=0.0,
    )
    imcube = np.zeros((nT, ny, nx))
    if (not isNone(im_mask)) and im_mask.ndim == 3:
        for Ti, uni_ang in enumerate(uni_angs):
            imcube[parangs == uni_ang] = convolve_with_spatial_psfs(
                inp_rot_uni[Ti],
                psfs,
                psf_inds=psf_inds[Ti],
                use_gpu=use_gpu,
                im_mask=im_mask[Ti],
            )
    else:
        for Ti, uni_ang in enumerate(uni_angs):
            imcube[parangs == uni_ang] = convolve_with_spatial_psfs(
                inp_rot_uni[Ti],
                psfs,
                psf_inds=psf_inds,
                use_gpu=use_gpu,
                im_mask=im_mask,
            )
    return imcube


def convolve_with_spatial_psfs(im0, psfs, psf_inds, im_mask=None, use_gpu=False):
    """
    Convolve an image with a spatially varying PSF map.

    Each pixel of ``im0`` is convolved with the nearest PSF slice from
    ``psfs``, as indicated by ``psf_inds``.

    Parameters
    ----------
    im0 : ndarray, shape (ny, nx)
        Input image.
    psfs : ndarray, shape (n_psf, ny, nx)
        Stack of PSF images.
    psf_inds : ndarray, shape (ny, nx)
        Map of indices into ``psfs`` selecting the PSF per pixel.
    im_mask : ndarray, optional
        Coronagraph throughput map multiplied into ``im0`` before convolution.
    use_gpu : bool, default False
        Use CuPy-based convolution.

    Returns
    -------
    imcon : ndarray, shape (ny, nx)
        Convolved image.
    """
    im = im0.copy()
    if not isNone(im_mask):
        im *= im_mask

    convolution_fn = psf_convolve_gpu if use_gpu else psf_convolve_cpu

    yi, xi = np.indices(im.shape)
    nonzero = im != 0.0

    psf_yhw, psf_xhw = np.ceil(np.array(psfs.shape[-2:]) / 2.0).astype(int)

    xi_nz, yi_nz = xi[nonzero], yi[nonzero]
    x1 = int(max(xi_nz.min() - psf_xhw, 0))
    x2 = int(min(xi_nz.max() + psf_xhw, im.shape[-1]))
    y1 = int(max(yi_nz.min() - psf_yhw, 0))
    y2 = int(min(yi_nz.max() + psf_yhw, im.shape[-2]))

    im_crop = im[y1 : y2 + 1, x1 : x2 + 1]
    psf_inds_crop = psf_inds[y1 : y2 + 1, x1 : x2 + 1]

    imcon_crop = np.zeros(im_crop.shape, dtype=im.dtype)
    for i in np.unique(psf_inds_crop):
        msk_i = psf_inds_crop == i
        im_to_convolve = np.where(msk_i, im_crop, 0.0)
        imcon_crop += convolution_fn(im_to_convolve, psfs[i])

    imcon = np.zeros_like(im)
    imcon[y1 : y2 + 1, x1 : x2 + 1] = imcon_crop
    return imcon


def psf_convolve_gpu(im, psf_im):
    """
    Convolve on GPU using CuPy's FFT-based convolution.

    Parameters
    ----------
    im : ndarray
        Image to convolve.
    psf_im : ndarray
        PSF kernel.

    Returns
    -------
    ndarray
        Convolved image, same shape as ``im``.
    """
    imcon = cp.asnumpy(cp_signal.fftconvolve(cp.array(im), cp.array(psf_im), mode="same"))
    return imcon


def psf_convolve_cpu(im, psf_im):
    """
    Convolve on CPU using SciPy's FFT-based convolution.

    Parameters
    ----------
    im : ndarray
        Image to convolve.
    psf_im : ndarray
        PSF kernel.

    Returns
    -------
    ndarray
        Convolved image, same shape as ``im``.
    """
    imcon = signal.fftconvolve(im, psf_im, mode="same")
    return imcon


def c_to_c_osamp(center, osamp):
    """
    Convert a Cartesian coordinate to the corresponding oversampled value.

    This uses typical zero-based indexing conventions.

    Parameters
    ----------
    center : array_like
        Coordinate(s) to convert.
    osamp : float
        Oversampling factor.

    Returns
    -------
    ndarray
        Converted coordinate(s).
    """
    return np.asarray(center) * osamp + 0.5 * (osamp - 1)


def generate_nircam_psf_grid(
    inst,
    coron_cents,
    source_spectrum=None,
    normalize=True,
    nr=6,
    ntheta=4,
    log_rscale=True,
    rmax=4,
    use_coeff=True,
):
    """
    Generate a grid of synthetic NIRCam PSFs and associated index/throughput maps.

    Parameters
    ----------
    inst : webbpsf_ext.webbpsf_ext_core.NIRCam_ext
        Configured NIRCam instrument object.
    coron_cents : ndarray, shape (n_rolls, 2)
        Detector coordinates ``[x, y]`` of the coronagraph center in each roll
        after registration.
    source_spectrum : pysynphot.Spectrum, optional
        Source spectrum used for PSF generation.
    normalize : bool, default True
        Normalize each occulted PSF to unit sum. If model images are multiplied
        by the coronagraph throughput map before convolution, the throughput is
        recovered at higher resolution.
    nr : int, default 6
        Number of radial PSF samples (excluding the origin).
    ntheta : int, default 4
        Number of azimuthal PSF samples.
    log_rscale : bool, default True
        If True, sample radii logarithmically between 0.01 and ``rmax`` arcsec.
    rmax : float, default 4
        Maximum separation (arcsec) from the coronagraph center.
    use_coeff : bool, default True
        Use coefficient-based PSF generation when available (``inst.calc_psf_from_coeff``).

    Returns
    -------
    psfs : ndarray, shape (n_samples, ny, nx)
        Stack of sampled PSFs.
    psf_inds_rolls : ndarray, shape (n_rolls, ny_os, nx_os)
        Map of nearest-PSF indices per (oversampled) detector pixel for each roll.
    im_mask_rolls : ndarray, shape (n_rolls, ny_os, nx_os)
        Coronagraph transmission maps per roll (oversampled).
    psf_offsets : ndarray, shape (2, n_samples)
        Offsets (``x``, ``y`` in arcsec) of each PSF sample.
    """

    if webbpsf_ext is None:
        raise RuntimeError("webbpsf_ext is required for PSF generation. Please install it.")

    siaf_ap = inst.siaf_ap
    osamp = inst.oversample

    nx, ny = siaf_ap.XSciSize, siaf_ap.YSciSize

    # Set up the grid
    if log_rscale:
        rvals = 10 ** (np.linspace(-2, np.log10(rmax), nr))
    else:
        rvals = np.linspace(0, rmax, nr + 1)[1:]

    thvals = np.linspace(0, 360, ntheta, endpoint=False)
    rvals_all = [0]
    thvals_all = [0]
    for r in rvals:
        for th in thvals:
            rvals_all.append(r)
            thvals_all.append(th)
    rvals_all = np.array(rvals_all)
    thvals_all = np.array(thvals_all)
    xgrid_off, ygrid_off = coords.rtheta_to_xy(rvals_all, thvals_all)  # Offset grid in arcsec

    rvals = np.unique(rvals_all)

    field_rot = 0 if inst._rotation is None else inst._rotation

    # Science positions in detector pixels
    xoff_sci_asec, yoff_sci_asec = coords.xy_rot(-1 * xgrid_off, -1 * ygrid_off, -1 * field_rot)

    psf_offsets = np.array([xoff_sci_asec, yoff_sci_asec])

    if use_coeff:
        psfs = inst.calc_psf_from_coeff(
            sp=source_spectrum,
            coord_vals=psf_offsets,
            coord_frame="idl",
            return_oversample=True,
            return_hdul=False,
            coron_rescale=True,
        )
    else:
        psfs = inst.calc_psf(
            sp=source_spectrum,
            coord_vals=psf_offsets,
            coord_frame="idl",
            return_oversample=True,
            return_hdul=False,
        )

    if normalize:
        psfs /= np.sum(psfs, axis=(-2, -1), keepdims=True)

    yg, xg = c_to_c_osamp(np.indices((ny * osamp, nx * osamp), dtype=np.float64), 1 / osamp)

    psf_inds_rolls = np.zeros((len(coron_cents), ny * osamp, nx * osamp), dtype=np.int32)
    im_mask_rolls = np.zeros(psf_inds_rolls.shape, dtype=np.float64)
    for j, cent in enumerate(coron_cents):
        im_mask_rolls[j] = generate_jwst_mask_image(inst, cent=cent, return_oversample=True)

        xmap_osamp, ymap_osamp = xg - cent[0], yg - cent[1]

        thvals_wrap0 = np.array([*thvals, *thvals])
        thvals_wrap = np.array([*thvals, *(thvals + 360.0)])

        rmap_osamp, tmap_osamp = coords.xy_to_rtheta(xmap_osamp, ymap_osamp)
        tmap_osamp = np.mod(tmap_osamp + 180.0, 360)

        rvals_px = rvals / inst.pixelscale

        nearest_rvals = rvals[np.argmin(np.array([np.abs(rmap_osamp - rval) for rval in rvals_px]), axis=0)]
        nearest_thvals = thvals_wrap0[np.argmin(np.array([np.abs(tmap_osamp - thval) for thval in thvals_wrap]), axis=0)]

        for i, (rval, thval) in enumerate(zip(rvals_all, thvals_all)):
            psf_inds_rolls[j, (nearest_rvals == rval) & (nearest_thvals == thval)] = i

    return psfs, psf_inds_rolls, im_mask_rolls, psf_offsets


def generate_jwst_mask_image(inst, cent, return_oversample=True):
    """
    Generate a JWST coronagraph mask transmission image.

    Parameters
    ----------
    inst : webbpsf_ext.webbpsf_ext_core.NIRCam_ext
        Instrument object.
    cent : array_like
        Center position ``[x, y]`` in oversampled pixels.
    return_oversample : bool, default True
        If True, return the oversampled mask image. If False, rebin to
        the detector sampling.

    Returns
    -------
    ndarray
        Transmission map (oversampled if requested).
    """
    osamp = inst.oversample
    im_mask_osamp = inst.gen_mask_image(
        npix=inst.siaf_ap.XSciSize * osamp, nd_squares=False, pixelscale=inst.pixelscale / osamp
    )
    im_mask_osamp = pad_or_crop_image(
        im_mask_osamp, new_size=im_mask_osamp.shape, new_cent=c_to_c_osamp(cent, osamp), cval=1
    )
    if return_oversample:
        return im_mask_osamp
    im_mask = webbpsf_ext.image_manip.frebin(im_mask_osamp, scale=1 / osamp, total=False)
    return im_mask


def propagate_nans_in_spatial_operation(
    a,
    fn,
    fn_args=None,
    fn_kwargs=None,
    fn_nan_kwargs=None,
    fn_zero_kwargs=None,
    prop_threshold=0,
    prop_zeros=True,
):
    """
    Apply a spatial operation while propagating NaNs (and optionally zeros).

    The operation is intentionally liberal with propagation: after rotation or
    resampling, more pixels may be marked as NaN/zero depending on
    ``prop_threshold``.

    Parameters
    ----------
    a : ndarray
        Input array.
    fn : callable
        Spatial operation (e.g., ``scipy.ndimage.rotate``).
    fn_args : list, optional
        Positional arguments for ``fn``.
    fn_kwargs : dict, optional
        Keyword arguments for ``fn`` when applied to data.
    fn_nan_kwargs : dict, optional
        Keyword arguments for ``fn`` when applied to the NaN mask.
    fn_zero_kwargs : dict, optional
        Keyword arguments for ``fn`` when applied to the zero mask.
    prop_threshold : float, default 0
        Mask values ``> prop_threshold`` are treated as True after transform.
    prop_zeros : bool, default True
        Also propagate zeros as a mask.

    Returns
    -------
    a_out : ndarray
        Transformed array with NaNs/zeros propagated through the operation.
    """
    if isNone(fn_args):
        fn_args = []
    if isNone(fn_kwargs):
        fn_kwargs = {}
    if isNone(fn_nan_kwargs):
        fn_nan_kwargs = fn_kwargs

    nans = np.isnan(a)
    any_nans = np.any(nans)

    if any_nans:
        a_out = fn(np.where(nans, 0.0, a), *fn_args, **fn_kwargs)
    else:
        a_out = fn(a, *fn_args, **fn_kwargs)

    if prop_zeros:
        zeros = a == 0.0
        any_zeros = np.any(zeros)
        if any_zeros:
            if isNone(fn_zero_kwargs):
                fn_zero_kwargs = fn_nan_kwargs
            zeros_out = fn(zeros.astype(float), *fn_args, **fn_zero_kwargs)
            a_out = np.where(zeros_out > prop_threshold, 0.0, a_out)
    if any_nans:
        nans_out = fn(nans.astype(float), *fn_args, **fn_nan_kwargs)
        a_out = np.where(nans_out > prop_threshold, np.nan, a_out)
    return a_out


def pad_or_crop_image(
    im,
    new_size,
    cent=None,
    new_cent=None,
    cval=np.nan,
    prop_threshold=1e-6,
    order=3,
    mode="constant",
    prefilter=True,
):
    """
    Pad or crop an image about a specified center.

    Parameters
    ----------
    im : ndarray
        Image to transform.
    new_size : tuple(int, int)
        New ``(ny, nx)`` shape.
    cent : array_like, optional
        Current center ``[x, y]``. If None, use geometric center.
    new_cent : array_like, optional
        Desired center ``[x, y]`` in the output.
    cval : float, default numpy.nan
        Fill value.
    prop_threshold : float, default 1e-6
        Threshold used when propagating masks.
    order : int, default 3
        Spline interpolation order for map_coordinates.
    mode : str, default "constant"
        Boundary mode for map_coordinates.
    prefilter : bool, default True
        Prefilter for spline interpolation.

    Returns
    -------
    ndarray
        Padded/cropped image.
    """
    new_size = np.asarray(new_size)
    im_size = np.array(im.shape)
    ny, nx = im_size

    if isNone(cent):
        cent = (np.array([nx, ny]) - 1.0) / 2.0

    if np.all([new_size == im_size, cent == new_cent]):
        return im.copy()

    im_out = propagate_nans_in_spatial_operation(
        im,
        pad_or_crop_about_pos,
        fn_args=[cent, new_size],
        fn_kwargs=dict(new_cent=new_cent, cval=cval, order=order, mode=mode, prefilter=prefilter),
        fn_nan_kwargs=dict(new_cent=new_cent, cval=cval, order=order, mode=mode, prefilter=False),
        prop_threshold=prop_threshold,
    )
    return im_out


def pad_or_crop_about_pos(
    im, pos, new_size, new_cent=None, cval=np.nan, order=3, mode="constant", prefilter=True
):
    """
    Pad or crop an image about a given position.

    Parameters
    ----------
    im : ndarray
        Input image or stack (..., ny, nx).
    pos : array_like
        Original center ``[x, y]``.
    new_size : tuple(int, int)
        Output size ``(ny, nx)``.
    new_cent : array_like, optional
        New center ``[x, y]`` in the output.
    cval : float, default numpy.nan
        Fill value.
    order : int, default 3
        Spline interpolation order.
    mode : str, default "constant"
        Boundary mode for map_coordinates.
    prefilter : bool, default True
        Prefilter for spline interpolation.

    Returns
    -------
    ndarray
        Padded/cropped image (same ndim as input).
    """
    ny, nx = im.shape[-2:]
    ny_new, nx_new = new_size
    if isNone(new_cent):
        new_cent = (np.array([nx_new, ny_new]) - 1.0) / 2.0

    nd = np.ndim(im)
    xg, yg = np.meshgrid(np.arange(nx_new, dtype=np.float64), np.arange(ny_new, dtype=np.float64))

    xg -= (new_cent[0] - pos[0])
    yg -= (new_cent[1] - pos[1])

    if nd == 2:
        im_out = ndimage.map_coordinates(
            im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter
        )
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = ndimage.map_coordinates(
                im_reshaped[i],
                np.array([yg, xg]),
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )
        im_out = im_out.reshape((*im.shape[:-2], ny, nx))
    return im_out


def dist_to_pt(pt, nx=201, ny=201, dtype=float):
    """
    Compute a distance map to a point.

    Parameters
    ----------
    pt : array_like
        Point ``[x, y]``.
    nx, ny : int, optional
        Output size. Defaults to 201.
    dtype : data-type, optional
        Output dtype. Default is ``float``.

    Returns
    -------
    ndarray, shape (ny, nx)
        Euclidean distance map.
    """
    xaxis = np.arange(0, nx, dtype=dtype) - pt[0]
    yaxis = np.arange(0, ny, dtype=dtype) - pt[1]
    return np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)


def rotate_image_cpu(im, angle, cent=None, new_cent=None, cval0=np.nan, prop_threshold=1e-6):
    """
    Rotate an image by ``angle`` (degrees) using CPU operations.

    Exact zeros are treated similarly to NaNs and can be propagated.

    Parameters
    ----------
    im : ndarray
        Input image or stack (..., ny, nx).
    angle : float
        Rotation angle in degrees.
    cent : array_like, optional
        Center of rotation ``[x, y]``. If None, use the geometric center.
    new_cent : array_like, optional
        New desired center in the output.
    cval0 : float, default numpy.nan
        Fill value.
    prop_threshold : float, default 1e-6
        Threshold for mask propagation.

    Returns
    -------
    ndarray
        Rotated image/stack.
    """
    if angle == 0.0:
        return im.copy()
    geom_cent = (np.array(im.shape[-2:][::-1]) - 1.0) / 2.0
    if isNone(cent) or np.all(cent == geom_cent):
        im_out = propagate_nans_in_spatial_operation(
            im,
            ndimage.rotate,
            fn_args=[angle],
            fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=cval0),
            fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
            prop_threshold=prop_threshold,
            prop_zeros=True,
        )
    else:
        im_out = propagate_nans_in_spatial_operation(
            im,
            rotate_about_pos,
            fn_args=[cent, angle],
            fn_kwargs=dict(cval=cval0, new_cent=new_cent),
            fn_nan_kwargs=dict(cval=0, prefilter=False),
            prop_threshold=prop_threshold,
            prop_zeros=True,
        )
    return im_out


def rotate_image_gpu(im0, angle, cent=None, new_cent=None, cval0=np.nan, prop_threshold=1e-6):
    """
    Rotate an image by ``angle`` (degrees) using GPU operations (CuPy).

    Parameters
    ----------
    im0 : ndarray
        Input image.
    angle : float
        Rotation angle in degrees.
    cent : array_like, optional
        Center of rotation ``[x, y]``. If None, use the geometric center.
    new_cent : array_like, optional
        New desired center in the output.
    cval0 : float, default numpy.nan
        Fill value.
    prop_threshold : float, default 1e-6
        Threshold for mask propagation.

    Returns
    -------
    ndarray
        Rotated image.
    """
    if angle == 0.0:
        return im0.copy()
    im = cp.asarray(im0)
    nans = cp.isnan(im)
    zeros = im == 0.0
    any_zeros = cp.any(zeros)
    any_nans = cp.any(nans)
    geom_cent = (np.array(im.shape[-2:][::-1]) - 1.0) / 2.0
    if isNone(cent) or np.all(cent == geom_cent):
        if any_nans:
            rot_im = cp_ndimage.rotate(cp.where(nans, 0.0, im), angle, axes=(-2, -1), reshape=False, cval=cval0)
        else:
            rot_im = cp_ndimage.rotate(im, angle, axes=(-2, -1), reshape=False, cval=cval0)
        if any_zeros:
            rot_zeros = cp_ndimage.rotate(zeros.astype(float), angle, axes=(-2, -1), prefilter=False, reshape=False)
            rot_im = cp.where(rot_zeros > prop_threshold, 0.0, rot_im)
        if any_nans:
            rot_nans = cp_ndimage.rotate(nans.astype(float), angle, axes=(-2, -1), prefilter=False, reshape=False)
            rot_im = cp.where(rot_nans > prop_threshold, cp.nan, rot_im)
    else:
        if any_nans:
            rot_im = rotate_about_pos_gpu(cp.where(nans, 0.0, im), cent, angle, cval=cval0, new_cent=new_cent)
        else:
            rot_im = rotate_about_pos_gpu(im, cent, angle, cval=cval0, new_cent=new_cent)
        if any_zeros:
            rot_zeros = rotate_about_pos_gpu(zeros.astype(float), cent, angle, prefilter=False, new_cent=new_cent)
            rot_im = cp.where(rot_zeros > prop_threshold, 0.0, rot_im)
        if any_nans:
            rot_nans = rotate_about_pos_gpu(nans.astype(float), cent, angle, prefilter=False, new_cent=new_cent)
            rot_im = cp.where(rot_nans > prop_threshold, cp.nan, rot_im)
    return cp.asnumpy(rot_im)


def rotate_about_pos_gpu(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode="constant", prefilter=True):
    """
    Rotate about an arbitrary position on GPU.

    Parameters
    ----------
    im : cupy.ndarray
        Input image or stack.
    pos : array_like
        Center of rotation ``[x, y]``.
    angle : float
        Rotation angle in degrees.
    new_cent : array_like, optional
        New desired center in the output.
    cval : float, default numpy.nan
        Fill value.
    order : int, default 3
        Spline interpolation order.
    mode : str, default "constant"
        Boundary mode for map_coordinates.
    prefilter : bool, default True
        Prefilter for spline interpolation.

    Returns
    -------
    cupy.ndarray
        Rotated image/stack.
    """
    ny, nx = im.shape[-2:]
    nd = cp.ndim(im)
    xg0, yg0 = cp.meshgrid(cp.arange(nx, dtype=cp.float64), cp.arange(ny, dtype=cp.float64))

    if not isNone(new_cent):
        xg0 -= (new_cent[0] - pos[0])
        yg0 -= (new_cent[1] - pos[1])

    xg, yg = xy_polar_ang_displacement_gpu(xg0 - pos[0], yg0 - pos[1], angle)
    xg += pos[0]
    yg += pos[1]

    if nd == 2:
        im_rot = cp_ndimage.map_coordinates(
            im, cp.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter
        )
    else:
        nI = int(cp.prod(cp.array(im.shape[:-2])))
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = cp.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = cp_ndimage.map_coordinates(
                im_reshaped[i], cp.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter
            )
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    xg, yg, xg0, yg0 = free_gpu(xg, yg, xg0, yg0)
    return im_rot


def rotate_about_pos(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode="constant", prefilter=True):
    """
    Rotate about an arbitrary position on CPU.

    Parameters
    ----------
    im : ndarray
        Input image or stack (..., ny, nx).
    pos : array_like
        Center of rotation ``[x, y]``.
    angle : float
        Rotation angle in degrees.
    new_cent : array_like, optional
        New desired center in the output.
    cval : float, default numpy.nan
        Fill value.
    order : int, default 3
        Spline interpolation order.
    mode : str, default "constant"
        Boundary mode for map_coordinates.
    prefilter : bool, default True
        Prefilter for spline interpolation.

    Returns
    -------
    ndarray
        Rotated image/stack.
    """
    ny, nx = im.shape[-2:]
    nd = np.ndim(im)
    yg0, xg0 = np.indices((ny, nx), dtype=np.float64)

    if not isNone(new_cent):
        xg0 -= (new_cent[0] - pos[0])
        yg0 -= (new_cent[1] - pos[1])

    xg, yg = xy_polar_ang_displacement(xg0 - pos[0], yg0 - pos[1], angle)
    xg += pos[0]
    yg += pos[1]

    if nd == 2:
        im_rot = ndimage.map_coordinates(
            im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter
        )
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = ndimage.map_coordinates(
                im_reshaped[i],
                np.array([yg, xg]),
                order=order,
                mode=mode,
                cval=cval,
                prefilter=prefilter,
            )
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    return im_rot


def rotate_hypercube(hcube, angles, cent=None, new_cent=None, ncores=-2, use_gpu=False, cval0=0.0):
    """
    Rotate an array of images where the last two axes are ``(y, x)``.

    Parameters
    ----------
    hcube : ndarray
        Hypercube of images. For a time series of images with shape (ny, nx),
        this should have shape (n_T, ny, nx). For a time series of IFS cubes
        with n_L wavelengths, shape (n_T, n_L, ny, nx).
    angles : ndarray, shape (n_T,)
        Rotation angles in degrees for the first dimension of ``hcube``.
    cent : array_like, optional
        Center of rotation ``[x, y]``.
    new_cent : array_like, optional
        Desired center in the output.
    ncores : int, default -2
        CPU threads for rotation when ``use_gpu=False``; ``-2`` uses all but one.
    use_gpu : bool, default False
        Use CuPy-based rotation.
    cval0 : float, default 0.0
        Fill value.

    Returns
    -------
    ndarray
        Rotated hypercube with the same shape as ``hcube``.
    """
    if use_gpu:
        rot_hcube = np.stack(
            [rotate_image_gpu(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent) for imcube, angle in zip(hcube, angles)]
        )
    else:
        rot_hcube = np.stack(
            Parallel(n_jobs=ncores, prefer="threads")(
                delayed(rotate_image_cpu)(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent)
                for imcube, angle in zip(hcube, angles)
            )
        )
    return rot_hcube


def pad_and_rotate_hypercube(hcube, angles, cent=None, ncores=-2, use_gpu=False, cval0=np.nan):
    """
    Pad images to avoid pixel loss, then rotate a hypercube.

    Parameters
    ----------
    hcube : ndarray
        Hypercube to rotate.
    angles : ndarray, shape (n_T,)
        Rotation angles in degrees.
    cent : array_like, optional
        Center of rotation ``[x, y]``. If None, uses geometric center.
    ncores : int, default -2
        CPU threads for rotation when ``use_gpu=False``.
    use_gpu : bool, default False
        Use CuPy-based rotation.
    cval0 : float, default numpy.nan
        Fill value.

    Returns
    -------
    hcube_pad_rot : ndarray
        Padded and rotated hypercube.
    cent_pad : ndarray
        New center after padding.
    """
    ny, nx = hcube.shape[-2:]
    if isNone(cent):
        cent = (np.array([nx, ny]) - 1.0) / 2.0
    dxmin, dxmax = np.array([0, nx]) - cent[0]
    dymin, dymax = np.array([0, ny]) - cent[1]
    corner_coords = np.array([[dxmax, dymax], [dxmax, dymin], [dxmin, dymin], [dxmin, dymax]])
    uni_angs = np.unique(angles)
    derot_corner_coords = np.vstack(
        [np.array(xy_polar_ang_displacement(*corner_coords.T, -ang)).T for ang in uni_angs]
    )
    dxmin_pad, dymin_pad = (np.ceil(np.abs(np.min(derot_corner_coords, axis=0) - np.array([dxmin, dymin])))).astype(int)
    dxmax_pad, dymax_pad = (np.ceil(np.abs(np.max(derot_corner_coords, axis=0) - np.array([dxmax, dymax])))).astype(int)
    hcube_pad = np.pad(
        hcube.copy(),
        [*[[0, 0] for _ in range(hcube.ndim - 2)], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]],
        constant_values=np.nan,
    )
    cent_pad = cent + np.array([dxmin_pad, dymin_pad])
    hcube_pad_rot = rotate_hypercube(hcube_pad, angles, cent=cent_pad, ncores=ncores, use_gpu=use_gpu, cval0=cval0)
    return hcube_pad_rot, cent_pad


def xy_polar_ang_displacement(x, y, dtheta):
    """
    Rotate Cartesian coordinates by an angle about the origin.

    Parameters
    ----------
    x, y : array_like
        Coordinates to rotate.
    dtheta : float
        Rotation angle in degrees.

    Returns
    -------
    newx, newy : ndarray
        Rotated coordinates.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.rad2deg(np.arctan2(y, x))
    new_theta = np.deg2rad(theta + dtheta)
    newx, newy = r * np.cos(new_theta), r * np.sin(new_theta)
    return newx, newy


def xy_polar_ang_displacement_gpu(x, y, dtheta):
    """
    GPU version of :func:`xy_polar_ang_displacement`.

    Parameters
    ----------
    x, y : cupy.ndarray
        Coordinates to rotate.
    dtheta : float
        Rotation angle in degrees.

    Returns
    -------
    cupy.ndarray
        Rotated coordinates ``(newx, newy)``.
    """
    r = cp.sqrt(x**2 + y**2)
    theta = cp.rad2deg(cp.arctan2(y, x))
    new_theta = cp.deg2rad(theta + dtheta)
    newx, newy = r * cp.cos(new_theta), r * cp.sin(new_theta)
    return newx, newy


def free_gpu(*args):
    """
    Free intermediate GPU buffers.

    Parameters
    ----------
    *args :
        Arrays to dereference before clearing the CuPy memory pools.

    Returns
    -------
    list or None
        Returns the list of inputs replaced by ``None`` (or ``None`` if only one).
    """
    N = len(args)
    args = list(args)
    for i in range(N):
        args[i] = None
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    if N <= 1:
        return None
    return args


def isNone(arg):
    """
    Convenience: robust None check that works with numpy arrays.

    Parameters
    ----------
    arg : Any
        Object to test.

    Returns
    -------
    bool
        True if ``arg`` is ``None``.
    """
    return isinstance(arg, type(None))


def mpl_stcen_extent(im, cent=None, pixelscale=None):
    """
    Compute a Matplotlib ``extent`` centered on a given star position.

    Parameters
    ----------
    im : ndarray
        Image.
    cent : array_like, optional
        Center coordinate ``[x, y]``. If None, use geometric center.
    pixelscale : float, optional
        Pixel scale to convert to physical units.

    Returns
    -------
    ndarray, shape (4,)
        Extent array: ``[xmin, xmax, ymin, ymax]``.
    """
    ny, nx = im.shape
    if isNone(cent):
        cent = (np.array([nx, ny]) - 1) / 2.0

    extent = np.array([0 - cent[0] - 0.5, nx - cent[0] - 0.5, 0 - cent[1] - 0.5, ny - cent[1] - 0.5])
    if not isNone(pixelscale):
        extent *= pixelscale
    return extent


# Optional GPU back-end (CuPy)
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy import signal as cp_signal

    use_gpu = True
    gpu = cp.cuda.Device(0)
    print("CuPy successfully imported. Using GPU where applicable. "
          "Set use_gpu=False to override this functionality.")
except ModuleNotFoundError:
    use_gpu = False
    if webbpsf_ext != None:
        print("Could not import CuPy. "
            "Setting: use_gpu=False (i.e., using CPU operations).")
