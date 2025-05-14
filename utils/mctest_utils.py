import numpy as np
from utils.SLD_utils import *
import matplotlib.pyplot as plt


###############################################################################
# Helper Methods

def create_empirical_err_map(data, annulus_width=5, mask_rad=9, outlier_pixels=None):    
    y,x = np.indices(data.shape)
    y -= data.shape[0]//2
    x -= data.shape[1]//2 
    radii = np.sqrt(x**2 + y**2) 
    noise_array = np.zeros_like(data)
    for i in range(0, int(np.max(radii)//annulus_width) ): 
        indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
        noise_array[indices] = np.nanstd(data[indices])
    mask = radii <= mask_rad
    noise_array[mask] = 0

    if(outlier_pixels != None):
        for pixel in outlier_pixels:
            noise_array[pixel[0]][pixel[1]] = noise_array[pixel[0]][pixel[1]] * 1e6 

    return noise_array

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

def get_inc_bounded_knots(inclination, radius, buffer = 0, num_knots=-1):
    if(num_knots <= 0):
        if(radius < 50):
            num_knots = 4
        else:
            num_knots = 6
    return jnp.linspace(jnp.cos(jnp.deg2rad(90-inclination-buffer)), jnp.cos(jnp.deg2rad(90+inclination+buffer)), num_knots)

def transpose_even_to_odd_spline(knot_vals, knots, new_knots):
    x_vals = jnp.linspace(-1, 1, 100)
    spline = InterpolatedUnivariateSpline_SPF.init(knot_vals, knots=knots)
    y_vals = spline(x_vals)

    new_k_vals = []
    for nk in new_knots:
        new_k_vals.append(y_vals[int((nk*100 + 100)/2)])
    return jnp.array(new_k_vals)

def plot_mc_img(name, target_image, err_map, sc_image, mc_image, init_val, fin_val):

    ## Get a good scaling
    y, x = np.indices(target_image.shape)
    y -= 70
    x -= 70 
    rads = np.sqrt(x**2+y**2)
    mask = (rads > 12)

    vmin = np.nanpercentile(target_image[mask], 1)
    vmax = np.nanpercentile(target_image[mask], 99.9)

    fig, axes = plt.subplots(3,3, figsize=(20,10))

    plt.title(name)

    # 1st column is miscellanious
    im = axes[0][0].imshow(target_image, origin='lower', cmap='inferno')
    axes[0][0].set_title("Target Image")
    plt.colorbar(im, ax=axes[0][0], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][0].imshow(err_map, origin='lower', cmap='inferno')
    axes[1][0].set_title("Error Map")
    plt.colorbar(im, ax=axes[1][0], shrink=0.75)
    im.set_clim(vmin, vmax)

    snr = target_image/err_map
    svmin = np.nanpercentile(snr[mask], 1)
    svmax = np.nanpercentile(snr[mask], 99.9)
    im = axes[2][0].imshow(snr, origin='lower', cmap='inferno')
    axes[2][0].set_title("SNR Ratio")
    plt.colorbar(im, ax=axes[2][0], shrink=0.75)
    im.set_clim(svmin, svmax)


    # 2nd column is scipy fitted results
    im = axes[0][1].imshow(sc_image, origin='lower', cmap='inferno')
    axes[0][1].set_title("Scipy Fitted Image: " + str(init_val))
    plt.colorbar(im, ax=axes[0][1], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][1].imshow(target_image-sc_image, origin='lower', cmap='inferno')
    axes[1][1].set_title("Scipy Residual")
    plt.colorbar(im, ax=axes[1][1], shrink=0.75)
    im.set_clim(vmin, vmax)

    snr = (target_image-sc_image)/err_map
    svmin = np.nanpercentile(snr[mask], 1)
    svmax = np.nanpercentile(snr[mask], 99.9)
    im = axes[2][1].imshow(snr, origin='lower', cmap='inferno')
    axes[2][1].set_title("Scipy Residual / Noise")
    plt.colorbar(im, ax=axes[2][1], shrink=0.75)
    im.set_clim(svmin, svmax)

    # 3rd column is MCMC results
    im = axes[0][2].imshow(mc_image, origin='lower', cmap='inferno')
    axes[0][2].set_title("MCMC Fitted Image: " + str(fin_val))
    plt.colorbar(im, ax=axes[0][2], shrink=0.75)
    im.set_clim(vmin, vmax)

    im = axes[1][2].imshow(target_image-mc_image, origin='lower', cmap='inferno')
    axes[1][2].set_title("MCMC Residual")
    plt.colorbar(im, ax=axes[1][2], shrink=0.75)
    im.set_clim(vmin, vmax)

    snr = (target_image-mc_image)/err_map
    svmin = np.nanpercentile(snr[mask], 1)
    svmax = np.nanpercentile(snr[mask], 99)
    im = axes[2][2].imshow(snr, origin='lower', cmap='inferno')
    axes[2][2].set_title("MCMC Residual / Noise")
    plt.colorbar(im, ax=axes[2][2], shrink=0.75)
    im.set_clim(svmin, svmax)


def get_aic(pos_log_likelihood, num_params):
    return 2 * pos_log_likelihood + 2 * num_params

def get_bic(pos_log_likelihood, num_params):
    return 2 * pos_log_likelihood + np.log(140*140) * num_params