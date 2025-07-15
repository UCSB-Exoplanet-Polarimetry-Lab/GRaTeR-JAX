from scipy import ndimage, signal
import numpy as np
import webbpsf_ext
from webbpsf_ext import image_manip, coords
from astropy.io import fits
from joblib import Parallel, delayed

webbpsf_ext.setup_logging('WARN')

def raw_model_to_convolved_model_cube(input_im, parangs, psfs, psf_inds, im_mask=None, cent=None, use_gpu=False, ncores=-2):
    """
    Creates a PSF-convolved sequence of model images at the indicated parangs (the parallactic angles of a data sequence).
    ___________
    Parameters:
    
        input_im: ndarray
            2D image array; the raw model image.
            
        parangs: ndarray
            1D array giving the parallactic angles (in degrees) to which input_im should be rotated.

        psfs: ndarray
            3D image array of spatially-sampled PSFs with which to convolve the model *after* rotation. If these PSFs are oversampled,
            the model image (input_im) should be oversampled by the same factor.
    
        psf_inds: ndarray
            2D or 3D array (with final two axes matching input_im; dtype=int); If 2D: each pixel value gives the index of the slice 
            in 'psfs' with which that pixel should be convolved. If 3D: as 2D, but with a different map for each roll angle (each unique
            value in parangs).
    
    Optional:
    
        im_mask: ndarray
            If psf_inds is 2D: 2D array of the coronagraph transmission (same shape as input_im), by which input_im will be multiplied before
            convolution. If psf_inds is 3D: the coronagraph transmission map for each roll angle of the data.
            
        cent: ndarray
            The cartesian pixel coordinate, ordered [x,y], corresponding to the star's location in input_im.
            
        use_gpu: bool
            If True, use faster GPU-based CuPy routines for rotation and convolution.
            
        ncores: int
            The number of processor cores to use for image rotation if use_gpu=False. Default value of -2 uses all but one available core.
            Ignored if use_gpu=True.
            
    Returns:
        imcube: ndarray
            3D array where each slice is the result of rotating input_im to the angle of the corresponding entry in parang,
            then convolving with the PSF.
    """
    nT = len(parangs)
    ny,nx = input_im.shape[-2:]
    uni_angs = np.unique(parangs)
        
    inp_rot_uni = rotate_hypercube(np.tile(input_im, (len(uni_angs),1,1)), uni_angs, cent=cent, ncores=ncores, use_gpu=use_gpu, cval0=0.)
    imcube = np.zeros((nT, ny, nx))
    if not isNone(im_mask) and im_mask.ndim == 3:
        for Ti, uni_ang in enumerate(uni_angs):
            imcube[parangs == uni_ang] = convolve_with_spatial_psfs(inp_rot_uni[Ti], psfs, psf_inds=psf_inds[Ti], use_gpu=use_gpu, im_mask=im_mask[Ti])
    else:
        for Ti, uni_ang in enumerate(uni_angs):
            imcube[parangs == uni_ang] = convolve_with_spatial_psfs(inp_rot_uni[Ti], psfs, psf_inds=psf_inds, use_gpu=use_gpu, im_mask=im_mask)
    return imcube


def convolve_with_spatial_psfs(im0, psfs, psf_inds, im_mask=None, use_gpu=False):
    """
    Creates a PSF-convolved image where each pixel of the input image has been convolved with the
    nearest spatially-sampled PSF. 
    
    Note: This can be sped up a little by preparing a boolean array where each slice is the shape
          of im0 and is True where the corresponding slice in psfs is the nearest match. However, 
          if `psfs' is very finely sampled, this would result in a very large array (e.g., if psfs
          samples at every pixel in im0, this would produce an array of shape (ny*nx, ny, nx)). 
          In testing, the time saved was marginal enough (~5%) that I've avoided this approach in
          favor of the more memory conscious inputs here.
    ___________
    Parameters:
    
        im0: ndarray
            2D image array to be convolved.

        psfs: ndarray
            3D image array of spatially-sampled PSFs with which to convolve im0. Generally, each
    
        psf_inds: ndarray
            2D array (same shape as im0; dtype=int); each entry gives the index of the slice in psfs with 
            which that pixel in im0 should be convolved.
    
    Optional:
    
        im_mask: ndarray
            2D array of the coronagraph throughput (same shape as im0), by which im0 will be multiplied before
            convolution.
        
        use_gpu: bool
            If True, use faster GPU-based CuPy routines for convolution.
            
    Returns:
        imcon: ndarray
            Convolved image of the same shape as im0.
    """
    im = im0.copy()
    if not isNone(im_mask):
        im *= im_mask

    convolution_fn = psf_convolve_gpu if use_gpu else psf_convolve_cpu
    
    yi,xi = np.indices(im.shape)
    nonzero = im != 0.
    
    psf_yhw, psf_xhw = np.ceil(np.array(psfs.shape[-2:])/2.).astype(int)

    xi_nz, yi_nz = xi[nonzero], yi[nonzero]
    x1, x2 = int(max(xi_nz.min()-psf_xhw, 0)), int(min(xi_nz.max()+psf_xhw, im.shape[-1]))
    y1, y2 = int(max(yi_nz.min()-psf_yhw, 0)), int(min(yi_nz.max()+psf_yhw, im.shape[-2]))
    
    im_crop = im[y1:y2+1, x1:x2+1]
    psf_inds_crop = psf_inds[y1:y2+1, x1:x2+1]
    
    imcon_crop = np.zeros(im_crop.shape, dtype=im.dtype)
    for i in np.unique(psf_inds_crop):
        msk_i = psf_inds_crop==i
        im_to_convolve = np.where(msk_i, im_crop, 0.)
        imcon_crop += convolution_fn(im_to_convolve, psfs[i])
        
    imcon = np.zeros_like(im)
    imcon[y1:y2+1, x1:x2+1] = imcon_crop
    return imcon


def psf_convolve_gpu(im, psf_im):
    """
    GPU-based PSF convolution using CuPy's version of scipy.signal's fast fourier transform.
    """
    imcon = cp.asnumpy(cp_signal.fftconvolve(cp.array(im), cp.array(psf_im), mode='same'))
    return imcon


def psf_convolve_cpu(im, psf_im):
    """
    CPU-based PSF convolution using scipy.signal's fast fourier transform.
    """
    imcon = signal.fftconvolve(im, psf_im, mode='same')
    return imcon


def c_to_c_osamp(center, osamp):
    """
    Converts a cartesian coordinate to the correct oversampled value
    for typical Python / NumPy zero-based indexing. 
    """
    return np.asarray(center)*osamp + 0.5*(osamp-1)


def generate_nircam_psf_grid(inst, coron_cents, source_spectrum=None, normalize=True, nr=6,
                             ntheta=4, log_rscale=True, rmax=4, use_coeff=True):
    """
    Creates a grid of synthetic PSFs using a WebbPSF-ext NIRCam instrument object.
    
    ___________
    Parameters:
    
        inst: webbpsf_ext.webbpsf_ext_core.NIRCam_ext
            NIRCam instrument object (set up appropriately) to use for generating PSFs.

        coron cents: ndarray
            2D array of shape (nrolls, 2) providing the detector position of the coronagraph
            in each roll of the target data following registration to align stellar positions.
    
    
    Optional:
    
        source_spectrum: pysynphot spectrum
            A pysynphot spectrum object to use for generating the PSFs.
            
        normalize: bool
            If True (default), all occulted PSFs will be normalized to sum to 1. By
            multiplying a model image by the coronagraphic throughput before convolution
            with these PSFs, the throughput is recovered at much higher resolution.
            
        nr: int
            The number of radial PSF samples to use in the grid. Actual grid will
            have nr+1 radial samples, since a grid point is added at r,theta = (0,0).
            
        ntheta: int
            The number of azimuthal PSF samples to use in the grid.
            
        log_rscale: bool
            If True (default), radial samples are generated in a logarithmic grid between
            0.01 and rmax arcseconds
            
        rmax: float
            The maximum radial separation in arcseconds from the coronagraph center to use 
            for the PSF grid.
        
        use_coeff: bool
            If True (default), uses coefficients to generate the PSF images. If False,
            computes PSFs directly using inst.calc_psf.
        
    Returns:
        psfs: ndarray
            3D stack of sampled synthetic PSF images
            
        psf_inds_rolls: ndarray
            A stack of maps indicating the index of the nearest PSF slice in 'psfs' to each
            (oversampled) detector pixel for each roll angle.
            
        im_mask_rolls: ndarray
            A stack of transmission maps indicating coronagraphic throughput per pixel
            for each roll angle.
            
        psf_offsets: ndarray
            An array of shape (2,Nsamples) providing the (x,y) offset from the coronagraph
            center for each PSF sample in "psfs" in units of arcsec.
    """
    
    siaf_ap = inst.siaf_ap
    osamp = inst.oversample
    
    nx, ny = siaf_ap.XSciSize, siaf_ap.YSciSize
    
    # Set up the grid:
    if log_rscale:
        rvals = 10**(np.linspace(-2,np.log10(rmax),nr))
    else:
        rvals = np.linspace(0, rmax, nr+1)[1:]
        
    thvals = np.linspace(0, 360, ntheta, endpoint=False)
    rvals_all = [0]
    thvals_all = [0]
    for r in rvals:
        for th in thvals:
            rvals_all.append(r)
            thvals_all.append(th)
    rvals_all = np.array(rvals_all)
    thvals_all = np.array(thvals_all)
    xgrid_off, ygrid_off = coords.rtheta_to_xy(rvals_all, thvals_all) # Mask Offset grid positions in arcsec
    
    rvals = np.unique(rvals_all)
    
    field_rot = 0 if inst._rotation is None else inst._rotation

    # Science positions in detector pixels
    xoff_sci_asec, yoff_sci_asec = coords.xy_rot(-1*xgrid_off, -1*ygrid_off, -1*field_rot)

    psf_offsets = np.array([xoff_sci_asec, yoff_sci_asec])

    if use_coeff:
        psfs = inst.calc_psf_from_coeff(sp=source_spectrum, coord_vals=psf_offsets, coord_frame='idl', return_oversample=True, return_hdul=False, coron_rescale=True)
    else:
        psfs = inst.calc_psf(sp=source_spectrum, coord_vals=psf_offsets, coord_frame='idl', return_oversample=True, return_hdul=False)

    if normalize:
        psfs /= np.sum(psfs, axis=(-2,-1), keepdims=True)

    yg, xg = c_to_c_osamp(np.indices((ny*osamp, nx*osamp), dtype=np.float64), 1/osamp)
    
    psf_inds_rolls = np.zeros((len(coron_cents), ny*osamp, nx*osamp), dtype=np.int32)
    im_mask_rolls = np.zeros(psf_inds_rolls.shape, dtype=np.float64)
    for j,cent in enumerate(coron_cents):
        im_mask_rolls[j] = generate_jwst_mask_image(inst, cent=cent, return_oversample=True)

        xmap_osamp, ymap_osamp = xg-cent[0], yg-cent[1]

        thvals_wrap0 = np.array([*thvals, *thvals])
        thvals_wrap = np.array([*thvals, *(thvals+360.)])

        rmap_osamp, tmap_osamp = coords.xy_to_rtheta(xmap_osamp, ymap_osamp)
        tmap_osamp = np.mod(tmap_osamp+180., 360)

        rvals_px = rvals/inst.pixelscale

        nearest_rvals = rvals[np.argmin(np.array([np.abs(rmap_osamp-rval) for rval in rvals_px]), axis=0)]
        nearest_thvals = thvals_wrap0[np.argmin(np.array([np.abs(tmap_osamp-thval) for thval in thvals_wrap]), axis=0)]

        for i,(rval,thval) in enumerate(zip(rvals_all, thvals_all)):
            psf_inds_rolls[j, (nearest_rvals == rval) & (nearest_thvals == thval)] = i
        
    return psfs, psf_inds_rolls, im_mask_rolls, psf_offsets


def generate_jwst_mask_image(inst, cent, return_oversample=True):
    osamp = inst.oversample
    im_mask_osamp = inst.gen_mask_image(npix=inst.siaf_ap.XSciSize*osamp, nd_squares=False, pixelscale=inst.pixelscale/osamp)
    im_mask_osamp = pad_or_crop_image(im_mask_osamp, new_size=im_mask_osamp.shape, new_cent=c_to_c_osamp(cent, osamp), cval=1)
    if return_oversample:
        return im_mask_osamp
    im_mask = webbpsf_ext.image_manip.frebin(im_mask_osamp, scale=1/osamp, total=False)
    return im_mask


def propagate_nans_in_spatial_operation(a, fn, fn_args=None, fn_kwargs=None, fn_nan_kwargs=None,
                                        fn_zero_kwargs=None, prop_threshold=0, prop_zeros=True):
    """
    This takes an array, a, and and a function that performs some spatial operation on a, fn,
    and attempts to propgate any nans (and optionally: zeros, which are often also non-physical values)
    through the indicated operation. Note: this operation is intentionally liberal with propgating the specified values.
    I.e., for rotation of an image with nans, expect there to be more NaN pixels following the operation. 
    This can be tuned somewhat by increasing the value of prop_threshold (0 <= prop_threshold <= 1)
    
    Example:

    import numpy as np
    from scipy import ndimage
    im = np.random.normal(loc=10, size=(101,101))
    im = ndimage.gaussian_filter(im, sigma=2.5)
    im[68:75, 34:48] = np.nan
    im[11:22, 8:19] = 0.
    angle = 30.0 # angle to rotate image by
    im_rot = propagate_nans_in_spatial_operation(im, ndimage.rotate, fn_args=[angle],
                                                 fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=np.nan, prefilter=False),
                                                 fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
                                                 prop_threshold=0, prop_zeros=True)
    """
    if isNone(fn_args): fn_args = []
    if isNone(fn_kwargs): fn_kwargs = {}
    if isNone(fn_nan_kwargs): fn_nan_kwargs = fn_kwargs
    
    nans = np.isnan(a)
    any_nans = np.any(nans)

    if any_nans:
        a_out = fn(np.where(nans, 0., a), *fn_args, **fn_kwargs)
    else: 
        a_out = fn(a, *fn_args, **fn_kwargs)
        
    if prop_zeros:
        zeros = a == 0.
        any_zeros = np.any(zeros)
        # Apply the operation to the boolean map of zeros 
        # >>> replace any locations > prop_threshold with zeros in the output
        if any_zeros:
            if isNone(fn_zero_kwargs):
                fn_zero_kwargs = fn_nan_kwargs
            zeros_out = fn(zeros.astype(float), *fn_args, **fn_zero_kwargs)
            a_out = np.where(zeros_out>prop_threshold, 0., a_out)
    if any_nans:
        nans_out = fn(nans.astype(float), *fn_args, **fn_nan_kwargs)
        a_out = np.where(nans_out>prop_threshold, np.nan, a_out)
    return a_out


def pad_or_crop_image(im, new_size, cent=None, new_cent=None, cval=np.nan, prop_threshold=1e-6, order=3, mode='constant', prefilter=True):
    new_size = np.asarray(new_size)
    im_size = np.array(im.shape)
    ny, nx = im_size
    
    if isNone(cent):
        cent = (np.array([nx,ny])-1.)/2.
        
    if np.all([new_size == im_size, cent == new_cent]):
        return im.copy()
    
    im_out = propagate_nans_in_spatial_operation(im, pad_or_crop_about_pos,
                                                 fn_args=[cent, new_size],
                                                 fn_kwargs=dict(new_cent=new_cent, cval=cval,
                                                                order=order, mode=mode,
                                                                prefilter=prefilter),
                                                 fn_nan_kwargs=dict(new_cent=new_cent, cval=cval,
                                                                order=order, mode=mode,
                                                                prefilter=False),
                                                 prop_threshold=prop_threshold)
    return im_out


def pad_or_crop_about_pos(im, pos, new_size, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    ny_new, nx_new = new_size
    if isNone(new_cent):
        new_cent = (np.array([nx_new,ny_new])-1.)/2.
        
    nd = np.ndim(im)
    xg, yg = np.meshgrid(np.arange(nx_new, dtype=np.float64), np.arange(ny_new, dtype=np.float64))
    
    xg -= (new_cent[0]-pos[0])
    yg -= (new_cent[1]-pos[1])

    if nd == 2:
        im_out = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_out = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_out[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_out = im_out.reshape((*im.shape[:-2], ny, nx))
    return im_out


def dist_to_pt(pt, nx=201, ny=201, dtype=float):
    """
    Returns a square distance array of size (ny,nx), 
    where each pixel corresponds to the euclidean distance
    of that pixel from "pt".
    """
    xaxis = np.arange(0, nx, dtype=dtype)-pt[0]
    yaxis = np.arange(0, ny, dtype=dtype)-pt[1]
    return np.sqrt(xaxis**2 + yaxis[:, np.newaxis]**2)


def rotate_image_cpu(im, angle, cent=None, new_cent=None, cval0=np.nan, prop_threshold=1e-6):
    """
    Rotates im by angle "angle" in degrees using CPU operations. Avoids "mixing" exact zero values,
    which should functionally be treated as nans. If cent is provided, rotates about cent. 
    Otherwise, uses ndimage's rotate (which is a bit faster) to rotate about the geometric center.
    """
    if angle == 0.:
        return im.copy()
    geom_cent = (np.array(im.shape[-2:][::-1])-1.)/2.
    if isNone(cent) or np.all(cent == geom_cent):
        im_out = propagate_nans_in_spatial_operation(im, ndimage.rotate, fn_args=[angle],
                                                     fn_kwargs=dict(axes=(-2, -1), reshape=False, cval=cval0),
                                                     fn_nan_kwargs=dict(axes=(-2, -1), reshape=False, prefilter=False),
                                                     prop_threshold=prop_threshold, prop_zeros=True)
    else:
        im_out = propagate_nans_in_spatial_operation(im, rotate_about_pos, fn_args=[cent, angle],
                                                     fn_kwargs=dict(cval=cval0, new_cent=new_cent),
                                                     fn_nan_kwargs=dict(cval=0, prefilter=False),
                                                     prop_threshold=prop_threshold, prop_zeros=True)
    return im_out


def rotate_image_gpu(im0, angle, cent=None, new_cent=None, cval0=np.nan, prop_threshold=1e-6):
    """
    Rotates im0 by angle "angle" in degrees using GPU operations. Avoids "mixing" exact zero values, which should functionally be treated as nans.
    If cent is provided, rotates about cent. Otherwise, uses CuPy's version of scipy.ndimage's rotate (which is a bit faster) to rotate about the
    geometric center.
    """
    if angle == 0.:
        return im0.copy()
    im = cp.asarray(im0)
    nans = cp.isnan(im)
    zeros = im == 0.
    any_zeros = cp.any(zeros)
    any_nans = cp.any(nans)
    geom_cent = (np.array(im.shape[-2:][::-1])-1.)/2.
    if isNone(cent) or np.all(cent == geom_cent):
        if any_nans:
            rot_im = cp_ndimage.rotate(cp.where(nans, 0., im), angle, axes=(-2, -1), reshape=False, cval=cval0)
        else:
            rot_im = cp_ndimage.rotate(im, angle, axes=(-2, -1), reshape=False, cval=cval0)
        if any_zeros:
            rot_zeros = cp_ndimage.rotate(zeros.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_zeros>prop_threshold, 0., rot_im)
        if any_nans:
            rot_nans = cp_ndimage.rotate(nans.astype(float), angle, axes=(-2, -1),  prefilter=False, reshape=False)
            rot_im = cp.where(rot_nans>prop_threshold, cp.nan, rot_im)
    else:
        if any_nans:
            rot_im = rotate_about_pos_gpu(cp.where(nans, 0., im), cent, angle, cval=cval0, new_cent=new_cent)
        else:
            rot_im = rotate_about_pos_gpu(im, cent, angle, cval=cval0, new_cent=new_cent)
        if any_zeros:
            rot_zeros = rotate_about_pos_gpu(zeros.astype(float), cent, angle,  prefilter=False, new_cent=new_cent)
            rot_im = cp.where(rot_zeros>prop_threshold, 0., rot_im)
        if any_nans:
            rot_nans = rotate_about_pos_gpu(nans.astype(float), cent, angle,  prefilter=False, new_cent=new_cent)
            rot_im = cp.where(rot_nans>prop_threshold, cp.nan, rot_im)
    return cp.asnumpy(rot_im)


def rotate_about_pos_gpu(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = cp.ndim(im)
    xg0, yg0 = cp.meshgrid(cp.arange(nx, dtype=cp.float64), cp.arange(ny, dtype=cp.float64))
    
    if not isNone(new_cent):
        xg0 -= (new_cent[0]-pos[0])
        yg0 -= (new_cent[1]-pos[1])
    
    xg,yg = xy_polar_ang_displacement_gpu(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]
    
    if nd == 2:
        im_rot = cp_ndimage.map_coordinates(im, cp.array([yg,xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = int(cp.prod(cp.array(im.shape[:-2])))
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = cp.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = cp_ndimage.map_coordinates(im_reshaped[i], cp.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    xg, yg, xg0, yg0 = free_gpu(xg, yg, xg0, yg0)
    return im_rot


def rotate_about_pos(im, pos, angle, new_cent=None, cval=np.nan, order=3, mode='constant', prefilter=True):
    ny, nx = im.shape[-2:]
    nd = np.ndim(im)
    yg0, xg0 = np.indices((ny,nx), dtype=np.float64)
    
    if not isNone(new_cent):
        xg0 -= (new_cent[0]-pos[0])
        yg0 -= (new_cent[1]-pos[1])
    
    xg,yg = xy_polar_ang_displacement(xg0-pos[0], yg0-pos[1], angle)
    xg += pos[0]
    yg += pos[1]

    if nd == 2:
        im_rot = ndimage.map_coordinates(im, np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
    else:
        nI = np.product(im.shape[:-2])
        im_reshaped = im.reshape((nI, ny, nx))
        im_rot = np.zeros((nI, ny, nx), dtype=im.dtype)
        for i in range(nI):
            im_rot[i] = ndimage.map_coordinates(im_reshaped[i], np.array([yg, xg]), order=order, mode=mode, cval=cval, prefilter=prefilter)
        im_rot = im_rot.reshape((*im.shape[:-2], ny, nx))
    return im_rot


def rotate_hypercube(hcube, angles, cent=None, new_cent=None, ncores=-2, use_gpu=False, cval0=0.):
    """
    Rotates an N-dimensional array, 'hcube', where the final two axes are assumed to be cartesian y and x 
    and where 'angles' is an array of angles (in degrees) matching the length of the first dimension of hcube.
    
    E.g., for a sequence of nT images having shape (ny,nx), hcube should have shape (nT,ny,nx) and angles should have shape (nT,)
    
    For a sequence of nT IFS image cubes each having nL wavelength slices of shape (ny,nx), hcube should have shape (nT, nL, ny, nx).
    
    Using the "new_cent" keyword, the sequence can be simultaneously translated as well to reduce interpolations.
    """
    if use_gpu:
        rot_hcube = np.stack([rotate_image_gpu(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent) for imcube, angle in zip(hcube, angles)])
    else:
        rot_hcube = np.stack(Parallel(n_jobs=ncores, prefer='threads')(delayed(rotate_image_cpu)(imcube, angle, cval0=cval0, cent=cent, new_cent=new_cent) for imcube, angle in zip(hcube, angles)))
    return rot_hcube


def pad_and_rotate_hypercube(hcube, angles, cent=None, ncores=-2, use_gpu=False, cval0=np.nan):
    """
    Like rotate_hypercube, but pads the images first to avoid loss of pixels. Returns the rotated 
    hypercube and the new center of the padded hypercube.
    """
    ny, nx = hcube.shape[-2:]
    if isNone(cent):
        cent = (np.array([nx, ny])-1.)/2.
    dxmin, dxmax = np.array([0, nx]) - cent[0]
    dymin, dymax = np.array([0, ny]) - cent[1]
    corner_coords = np.array([[dxmax, dymax],
                              [dxmax, dymin],
                              [dxmin, dymin],
                              [dxmin, dymax]])
    uni_angs = np.unique(angles)
    derot_corner_coords = np.vstack([np.array(xy_polar_ang_displacement(*corner_coords.T, -ang)).T for ang in uni_angs])
    dxmin_pad, dymin_pad = (np.ceil(np.abs(np.min(derot_corner_coords, axis=0) - np.array([dxmin, dymin])))).astype(int)
    dxmax_pad, dymax_pad = (np.ceil(np.abs(np.max(derot_corner_coords, axis=0) - np.array([dxmax, dymax])))).astype(int)
    hcube_pad = np.pad(hcube.copy(), [*[[0,0] for i in range(hcube.ndim-2)], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]], constant_values=np.nan)
    cent_pad = cent + np.array([dxmin_pad, dymin_pad])
    hcube_pad_rot = rotate_hypercube(hcube_pad, angles, cent=cent_pad, ncores=ncores, use_gpu=use_gpu, cval0=cval0)
    return hcube_pad_rot, cent_pad


def xy_polar_ang_displacement(x, y, dtheta):
    """
    Rotates cartesian coordinates x and y by angle dtheta (deg) about (0,0).
    """
    r = np.sqrt(x**2+y**2)
    theta = np.rad2deg(np.arctan2(y,x))
    new_theta = np.deg2rad(theta+dtheta)
    newx,newy = r*np.cos(new_theta),r*np.sin(new_theta)
    return newx,newy


def xy_polar_ang_displacement_gpu(x, y, dtheta):
    r = cp.sqrt(x**2+y**2)
    theta = cp.rad2deg(cp.arctan2(y,x))
    new_theta = cp.deg2rad(theta+dtheta)
    newx,newy = r*cp.cos(new_theta),r*cp.sin(new_theta)
    return newx,newy


def free_gpu(*args):
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
    Just a quick convenience/shorthand function.
    "if isNone(x)" works for any x, whereas "if x == None"
    will sometimes cause a crash (e.g., if x is a numpy array).
    """
    return isinstance(arg, type(None))


def mpl_stcen_extent(im, cent=None, pixelscale=None):
    ny, nx = im.shape
    if isNone(cent):
        cent = (np.array([nx,ny])-1)/2.
        
    extent = np.array([0-cent[0]-0.5,nx-cent[0]-0.5, 0-cent[1]-0.5,ny-cent[1]-0.5])
    if not isNone(pixelscale):
        extent *= pixelscale
    return extent

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from cupyx.scipy import signal as cp_signal
    use_gpu = True
    gpu = cp.cuda.Device(0)
    print("CuPy succesfully imported. Using GPU where applicable. "
           "Set use_gpu=False to override this functionality.")
except ModuleNotFoundError:
    use_gpu = False
    print("Could not import CuPy. "
          "Setting: use_gpu=False (i.e., using CPU operations).")