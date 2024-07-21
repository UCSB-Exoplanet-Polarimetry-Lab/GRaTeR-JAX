import jax
import jax.numpy as jnp
from SLD_ojax import ScatteredLightDisk
from SLD_utils import DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, GAUSSIAN_PSF
from functools import partial
import jax.scipy.signal as jss

@partial(jax.jit, static_argnums=(0,1,4))
def jax_model(DistrModel, FuncModel, disk_params, spf_params, PSFModel = None,
                  halfNbSlices=25, e=0., ksi0=3., gamma=2., beta=1.,
                  nx=140, ny=140, pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params['alpha_in'], aout=disk_params['alpha_out'], a=disk_params['sma'],
                                   e=e, ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params['inclination'], disk_params['position_angle'],
                                              disk_params['alpha_in'], disk_params['alpha_out'], disk_params['sma'],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    func_params = FuncModel.pack_pars(spf_params)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        disk_image_psf = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return disk_params['flux_scaling']*scattered_light_image


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle
@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling, PSFModel = None,
                halfNbSlices=25, ksi0=3., gamma=2., beta=1.,
                nx=140, ny=140, pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   e=0., ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    func_params = FuncModel.pack_pars(spf_params)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')

    return flux_scaling*scattered_light_image


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle
@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_all_1d(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                     PSFModel = None, halfNbSlices=25, ksi0=3., gamma=2., beta=1.,
                    nx=140, ny=140, pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   e=0., ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params,
                                                                FuncModel, x_vector, y_vector, scattered_light_map, image, limage,
                                                                tmp, halfNbSlices)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return flux_scaling*scattered_light_image


@partial(jax.jit, static_argnums=(0,1,4))
def jax_model_spline(DistrModel, FuncModel, disk_params, spf_params, PSFModel = None,
                  halfNbSlices=25, ksi0=3., gamma=2., beta=1.,
                  nx=140, ny=140, pxInArcsec=0.01414, distance=50., knots=jnp.linspace(1,-1,6)):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params['alpha_in'], aout=disk_params['alpha_out'], a=disk_params['sma'],
                                   e=0., ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params['inclination'], disk_params['position_angle'],
                                              disk_params['alpha_in'], disk_params['alpha_out'], disk_params['sma'],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    func_params = FuncModel.pack_pars(spf_params, knots=knots)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return disk_params['flux_scaling']*scattered_light_image


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle
@partial(jax.jit, static_argnums=(0,1,7))
def jax_model_all_1d_cent(DistrModel, FuncModel, xc, yc, disk_params, spf_params, flux_scaling,
                     PSFModel = None, halfNbSlices=25, ksi0=3., gamma=2., beta=1.,
                     nx=140, ny=140,
                    pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   e=0., ksi0=ksi0, gamma=gamma, beta=beta, amin=0., dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    xcent = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    ycent = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xcent)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - ycent)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params,
                                                                FuncModel, x_vector, y_vector, scattered_light_map, image, limage,
                                                                tmp, halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - xc + xcent
    y = y - yc + ycent
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return flux_scaling*scattered_light_image


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7:amin 8: ksi, 9: gamma, 10: beta
@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_all_1d_full(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                     PSFModel = None, halfNbSlices=25, nx=140, ny=140, pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   amin=disk_params[7], ksi0=disk_params[8], gamma=disk_params[9], beta=disk_params[10], dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              nx=nx, ny=ny, distance = distance, omega =0., pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    xcent = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    ycent = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xcent)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - ycent)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params,
                                                                FuncModel, x_vector, y_vector, scattered_light_map, image, limage,
                                                                tmp, halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[5] + xcent
    y = y - disk_params[6] + ycent
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        psf_image /= jnp.nansum(psf_image)
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return flux_scaling*scattered_light_image

# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7:amin 8: ksi, 
# 9: gamma, 10: beta, 11: e, 12: omega
@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_all_1d_ecc(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                     PSFModel = None, halfNbSlices=25, nx=140, ny=140, pxInArcsec=0.01414, distance=50.):

    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   amin=disk_params[7], ksi0=disk_params[8], gamma=disk_params[9], beta=disk_params[10], 
                                   dens_at_r0=1.,e = disk_params[11], )
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              omega = disk_params[12],
                                              nx=nx, ny=ny, distance = distance, pxInArcsec=pxInArcsec,
                                              xdo=0., ydo=0.)

    xcent = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    ycent = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xcent)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - ycent)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = ScatteredLightDisk.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params,
                                                                FuncModel, x_vector, y_vector, scattered_light_map, image, limage,
                                                                tmp, halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[5] + xcent
    y = y - disk_params[6] + ycent
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if(PSFModel!=None):
        psf_image = PSFModel.generate(jnp.meshgrid(x_vector, y_vector))
        psf_image /= jnp.nansum(psf_image)
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return flux_scaling*scattered_light_image