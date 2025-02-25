import jax
import jax.numpy as jnp
from utils.SLD_ojax import ScatteredLightDisk
from utils.SLD_utils import DustEllipticalDistribution2PowerLaws, DoubleHenyeyGreenstein_SPF, GAUSSIAN_PSF
from functools import partial
import jax.scipy.signal as jss

## This file brings together the Scattered Light Disk models, dust distribution models, SPF models into one easier-to-use model function
## multiple definitions currently exist to enable different free parameters; future updates will have more flexible infrastructure for fitting

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


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: e, 8: omega
@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_all_1d_full(DistrModel, FuncModel, disk_params, spf_params, flux_scaling,
                     PSFModel = None, halfNbSlices=25, nx=140, ny=140, pxInArcsec=0.01414, distance=50.):
 
    distr_params = DistrModel.init(accuracy=5.e-3, ain=disk_params[0], aout=disk_params[1], a=disk_params[2],
                                   e=disk_params[7], ksi0=3, gamma=2, beta=1, amin=0, dens_at_r0=1.)
    disk_params_jax = ScatteredLightDisk.init(distr_params, disk_params[3], disk_params[4],
                                              disk_params[0], disk_params[1], disk_params[2],
                                              nx=nx, ny=ny, distance = distance, omega = disk_params[8],
                                              pxInArcsec=pxInArcsec, xdo=0., ydo=0.)

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
        scattered_light_image = jss.convolve2d(scattered_light_image, psf_image, mode='same')
    
    return flux_scaling*scattered_light_image


@partial(jax.jit, static_argnums=(0,1,5))
def jax_model_winnie(DistrModel, FuncModel, disk_params, spf_params, winnie_psf,
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

    convolved_image = jnp.mean(winnie_psf.get_convolved_cube(scattered_light_image), axis=0)
    
    return disk_params['flux_scaling']*convolved_image


# 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle
@partial(jax.jit, static_argnums=(0,1,6))
def jax_model_all_1d_cent_winnie(DistrModel, FuncModel, xc, yc, disk_params, spf_params, winnie_psf, flux_scaling,
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

    convolved_image = jnp.mean(winnie_psf.get_convolved_cube(scattered_light_image), axis=0)
    
    return flux_scaling*scattered_light_image