import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'PSFModel', 'StellarPSFModel', 'nx', 'ny', 'halfNbSlices'])
def jax_model(DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params,
              distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25, flux_scaling = 1e6):

    distr_params = DistrModel.init(accuracy=disk_params[0], alpha_in=disk_params[1], alpha_out=disk_params[2], sma=disk_params[3],
                                   e=disk_params[4], ksi0=disk_params[5], gamma=disk_params[6], beta=disk_params[7],
                                   rmin=disk_params[8], dens_at_r0=disk_params[9])
    disk_params_jax = DiskModel.init(distr_params, disk_params[10], disk_params[11],
                                              disk_params[1], disk_params[2], disk_params[3],
                                              nx=nx, ny=ny, distance = distance,
                                              omega = disk_params[15], pxInArcsec=pxInArcsec)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = DiskModel.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[12] + xc
    y = y - disk_params[13] + yc
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if PSFModel != None:
        scattered_light_image = PSFModel.generate(scattered_light_image, psf_params)

    scattered_light_image*flux_scaling

    if StellarPSFModel != None:
        scattered_light_image = scattered_light_image + StellarPSFModel.compute_stellar_psf_image(stellar_psf_params, nx, ny)

    return scattered_light_image


@partial(jax.jit, static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'winnie_psf', 'StellarPSFModel', 'nx', 'ny', 'halfNbSlices'])
def jax_model_winnie(DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params,
                     distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25, flux_scaling = 1e6):

    distr_params = DistrModel.init(accuracy=disk_params[0], alpha_in=disk_params[1], alpha_out=disk_params[2], sma=disk_params[3],
                                   e=disk_params[4], ksi0=disk_params[5], gamma=disk_params[6], beta=disk_params[7],
                                   rmin=disk_params[8], dens_at_r0=disk_params[9])
    disk_params_jax = DiskModel.init(distr_params, disk_params[10], disk_params[11],
                                              disk_params[1], disk_params[2], disk_params[3],
                                              nx=nx, ny=ny, distance = distance,
                                              omega = disk_params[15], pxInArcsec=pxInArcsec)

    yc, xc = ny, nx
    xc = jnp.where(nx%2==1, nx/2-0.5, nx/2).astype(int)
    yc = jnp.where(ny%2==1, ny/2-0.5, ny/2).astype(int)

    x_vector = (jnp.arange(0, nx) - xc)*pxInArcsec*distance
    y_vector = (jnp.arange(0, ny) - yc)*pxInArcsec*distance

    scattered_light_map = jnp.zeros((ny, nx))
    image = jnp.zeros((ny, nx))

    limage = jnp.zeros([2*halfNbSlices-1, ny, nx])
    tmp = jnp.arange(0, halfNbSlices)
    
    scattered_light_image = DiskModel.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, spf_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[12] + xc
    y = y - disk_params[13] + yc
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    scattered_light_image = jnp.mean(winnie_psf.get_convolved_cube(scattered_light_image), axis=0)

    scattered_light_image*flux_scaling

    if StellarPSFModel != None:
        scattered_light_image = scattered_light_image + StellarPSFModel.compute_stellar_psf_image(stellar_psf_params, nx, ny)

    return scattered_light_image


@partial(jax.jit, static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'PSFModel', 'StellarPSFModel', 'nx', 'ny', 'halfNbSlices'])
def jax_model_spline(DiskModel, DistrModel, FuncModel, PSFModel, StellarPSFModel, disk_params, spf_params, psf_params, stellar_psf_params,
                     distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25, flux_scaling = 1e6,
                     knots=jnp.linspace(1,-1,6)):

    distr_params = DistrModel.init(accuracy=disk_params[0], alpha_in=disk_params[1], alpha_out=disk_params[2], sma=disk_params[3],
                                   e=disk_params[4], ksi0=disk_params[5], gamma=disk_params[6], beta=disk_params[7],
                                   rmin=disk_params[8], dens_at_r0=disk_params[9])
    disk_params_jax = DiskModel.init(distr_params, disk_params[10], disk_params[11],
                                              disk_params[1], disk_params[2], disk_params[3],
                                              nx=nx, ny=ny, distance = distance,
                                              omega = disk_params[15], pxInArcsec=pxInArcsec)

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
    
    scattered_light_image = DiskModel.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)
    
    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[12] + xc
    y = y - disk_params[13] + yc
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    if PSFModel != None:
        scattered_light_image = PSFModel.generate(scattered_light_image, psf_params)

    scattered_light_image = scattered_light_image*flux_scaling

    if StellarPSFModel != None:
        scattered_light_image = scattered_light_image + StellarPSFModel.compute_stellar_psf_image(stellar_psf_params, nx, ny)

    return scattered_light_image


@partial(jax.jit, static_argnames=['DiskModel', 'DistrModel', 'FuncModel', 'winnie_psf', 'StellarPSFModel', 'nx', 'ny', 'halfNbSlices'])
def jax_model_spline_winnie(DiskModel, DistrModel, FuncModel, winnie_psf, StellarPSFModel, disk_params, spf_params, stellar_psf_params,
                     distance = 0., pxInArcsec = 0., nx = 140, ny = 140, halfNbSlices = 25,
                     flux_scaling = 1e6, knots=jnp.linspace(1,-1,6)):

    distr_params = DistrModel.init(accuracy=disk_params[0], alpha_in=disk_params[1], alpha_out=disk_params[2], sma=disk_params[3],
                                   e=disk_params[4], ksi0=disk_params[5], gamma=disk_params[6], beta=disk_params[7],
                                   rmin=disk_params[8], dens_at_r0=disk_params[9])
    disk_params_jax = DiskModel.init(distr_params, disk_params[10], disk_params[11],
                                              disk_params[1], disk_params[2], disk_params[3],
                                              nx=nx, ny=ny, distance = distance,
                                              omega = disk_params[15], pxInArcsec=pxInArcsec)

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
    
    scattered_light_image = DiskModel.compute_scattered_light_jax(disk_params_jax, distr_params, DistrModel, func_params, FuncModel,
                                                                  x_vector, y_vector, scattered_light_map, image, limage, tmp,
                                                                  halfNbSlices)

    dims = scattered_light_image.shape
    x, y = jnp.meshgrid(jnp.arange(dims[1], dtype=jnp.float32), jnp.arange(dims[0], dtype=jnp.float32))
    x = x - disk_params[12] + xc
    y = y - disk_params[13] + yc
    scattered_light_image = jax.scipy.ndimage.map_coordinates(jnp.copy(scattered_light_image),
                                                            jnp.array([y, x]),order=1,cval = 0.)

    scattered_light_image = jnp.mean(winnie_psf.get_convolved_cube(scattered_light_image), axis=0)

    scattered_light_image = scattered_light_image*flux_scaling

    if StellarPSFModel != None:
        scattered_light_image = scattered_light_image + StellarPSFModel.compute_stellar_psf_image(stellar_psf_params, nx, ny)

    return scattered_light_image
