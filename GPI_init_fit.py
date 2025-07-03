import os
os.environ["WEBBPSF_PATH"] = 'webbpsf-data'
os.environ["WEBBPSF_EXT_PATH"] = 'webbpsf-data'
os.environ["PYSYN_CDBS"] = "cdbs"
from astropy.io import fits
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.15'
jax.config.update("jax_enable_x64", True)
import pandas as pd
from optimization.optimize_framework import Optimizer, OptimizeUtils
from disk_model.objective_functions import objective_model, objective_ll, objective_fit, Parameter_Index
from disk_model.SLD_ojax import ScatteredLightDisk
from disk_model.SLD_utils import *

    
def init_fit(name,spf_type='spline',plot=True,save=False,mc=False,num_knots=6):
    fits_image_filepath = "/home/blewis/GPI_data/" + str(name) + ".fits"
    hdul = fits.open(fits_image_filepath)
    if '4796' in name:
        target_image = OptimizeUtils.process_image(hdul['SCI'].data[1,:,:],bounds=(50, 230, 50, 230))
        err_map = OptimizeUtils.process_image(OptimizeUtils.create_empirical_err_map(hdul['SCI'].data[2,:,:]),bounds=(50, 230, 50, 230))
        misc_params = Parameter_Index.misc_params
        misc_params['nx'] = 180
        misc_params['ny'] = 180 #, outlier_pixels=[(57, 68)]))
    else:
        target_image = OptimizeUtils.process_image(hdul['SCI'].data[1,:,:])
        err_map = OptimizeUtils.process_image(OptimizeUtils.create_empirical_err_map(hdul['SCI'].data[2,:,:])) #, outlier_pixels=[(57, 68)]))
        misc_params = Parameter_Index.misc_params
        misc_params['nx'] = 140
        misc_params['nx'] = 140

    if spf_type=='spline':
        spf_params = InterpolatedUnivariateSpline_SPF.params
        psf_params = EMP_PSF.params
        disk_params = Parameter_Index.disk_params

        image_data = pd.read_csv('image_info_filt.csv')
        image_data.set_index("Name", inplace=True)
        image_data.columns = ["Radius", "Inclination", "Position Angle", "Distance", "a_in", "a_out", "eccentricity", "ksi0", "gamma", "beta", "omega", "x_center", "y_center", "knots"]
        row = image_data.loc[name]

        disk_params['sma'] = row["Radius"]
        disk_params['inclination'] = row["Inclination"]
        disk_params['position_angle'] = row["Position Angle"]
        misc_params['distance'] = row["Distance"]
        spf_params['num_knots'] = num_knots
        spf_params['knot_values'] = jnp.full(spf_params['num_knots'],0.5)
        disk_params['alpha_in'] = row['a_in']
        disk_params['alpha_out'] = row['a_out']
        disk_params['e'] = row['eccentricity']
        disk_params['ksi0'] = row['ksi0']
        disk_params['gamma'] = row['gamma']
        disk_params['beta'] = row['beta']
        disk_params['omega'] = row['omega']
        disk_params['x_center'] = row['x_center']
        disk_params['y_center'] = row['y_center']

        opt = Optimizer(disk_params, spf_params, psf_params, misc_params, 
                ScatteredLightDisk, DustEllipticalDistribution2PowerLaws, InterpolatedUnivariateSpline_SPF, EMP_PSF)
        fit_keys = ['alpha_in', 'alpha_out', 'sma', 'e', 'ksi0','gamma','beta','omega','inclination', 'position_angle', 'x_center', 'y_center', 'flux_scaling','knot_values']
        opt.inc_bound_knots()
        opt.scale_initial_knots(target_image)
        soln = opt.scipy_optimize(fit_keys, target_image, err_map, disp_soln=True,iters = 1000)
        opt.scale_spline_to_fixed_point(0, 1)
        spf_soln = opt.spf_params
        optimal_image = opt.model()
        optimal_ll = opt.log_likelihood(target_image,err_map)
        knots = opt.spf_params['knot_values']
        angles = np.linspace(np.rad2deg(np.arccos(opt.spf_params['up_bound'])),np.rad2deg(np.arccos(opt.spf_params['low_bound'])),spf_params['num_knots'])
        titles = 'alpha_in, alpha_out, sma, e, ksi0, gamma, beta, omega, inclination, position_angle, x_center, y_center, flux, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11'
        if save==True:
            #np.savetxt('../GPI_results/{}_init_fit_params_{}knots.txt'.format(name,num_knots),soln.x,header=titles)
            #np.savetxt('../GPI_results/{}_init_fit_knotvals_{}knots.txt'.format(name,num_knots),knots,header=titles)
            #np.savetxt('../GPI_results/{}_init_fit_angles_{}knots.txt'.format(name,num_knots),angles,header=titles)
            pass
    if plot==True:
        fig, axes = plt.subplots(1,3, figsize=(20,10))
        mask = OptimizeUtils.get_mask(target_image)
        vmin = np.nanpercentile(target_image[mask], 1)
        vmax = np.nanpercentile(target_image[mask], 99)

        xmax=70
        extent = [-(xmax*14.1)/2000, (xmax*14.1)/2000, (xmax*14.1)/2000, -(xmax*14.1)/2000]

        for ax in axes:
            ax.tick_params(axis='both', which='major', labelsize=12)

        im = axes[0].imshow(target_image, origin='lower', cmap='inferno',extent=extent,vmin=vmin,vmax=vmax)
        axes[0].set_title("Data",fontsize=16)
        axes[0].set_ylabel('$\Delta$RA (arcsec)',fontsize=14)
        axes[0].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
        #plt.colorbar(im, ax=axes[0], shrink=0.5)
        im.set_clim(vmin, vmax)

        im = axes[1].imshow(optimal_image, origin='lower', cmap='inferno',extent=extent,vmin=vmin,vmax=vmax)
        axes[1].set_title("Model",fontsize=16)
        #plt.colorbar(im, ax=axes[1], shrink=0.5)
        im.set_clim(vmin, vmax)
        axes[1].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)

        im = axes[2].imshow(target_image-optimal_image, origin='lower', cmap='inferno',extent=extent,vmin=vmin,vmax=10)
        axes[2].set_title("Residuals",fontsize=16)
        im.set_clim(vmin, vmax)
        axes[2].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
        plt.tight_layout()
        cb = plt.colorbar(im, ax=axes, shrink=0.5,pad=0.01)
        cb.set_label('Arbitrary Flux Units',fontsize=14)#,rotation=270)
        cb.ax.tick_params(labelsize=12)
        print('I AM PLOTTING')
        plt.savefig('../GPI_results/{}_initial_fit_{}knots.png'.format(name,num_knots))

    #opt.print_params()
    if mc==False:
        return optimal_image, soln.x, optimal_ll, spf_soln

    if mc==True:
        if bounds==None:
            print("No bounds provided, using default bounds")
            print("we're warning you now -- if you changed the number of knots from the default, you need to change the bounds")
            spline_lb = np.log(0.1)
            spline_ub = np.log(5)
            bounds = [np.array([0.1, -15, 0, 0, -15, -15, -15, 0, 0, 0, 80, 80, 0, spline_lb, spline_lb, spline_lb, spline_lb, spline_lb, spline_lb]),
                    np.array([15, -0.1, 200, 1, 15, 15, 15, 360, 180, 360, 100, 100, 5e3, spline_ub, spline_ub, spline_ub, spline_ub, spline_ub, spline_ub])]
        mc_model = opt.mcmc(fit_keys, target_image, err_map, bounds, nwalkers=100, niter=100, burns=20)
        mc_soln = np.median(mc_model.sampler.flatchain, axis=0)
        img = opt.model()
        if save==True:
            np.savetxt('../GPI_results/{}_MCMC_fullchain_{}knots.txt'.format(name,num_knots),mc_model.sampler.flatchain,header=titles)
            np.savetxt('../GPI_results/{}_MCMC_bestfit_params_{}knots.txt'.format(name,num_knots),mc_soln,header=titles)
        if plot==True:
            fig, axes = plt.subplots(1,3, figsize=(20,10))
            mask = OptimizeUtils.get_mask(target_image)
            vmin = np.nanpercentile(target_image[mask], 1)
            vmax = np.nanpercentile(target_image[mask], 99)

            xmax=70
            extent = [-(xmax*14.1)/2000, (xmax*14.1)/2000, (xmax*14.1)/2000, -(xmax*14.1)/2000]

            for ax in axes:
                ax.tick_params(axis='both', which='major', labelsize=12)

            im = axes[0].imshow(target_image, origin='lower', cmap='inferno',extent=extent)
            axes[0].set_title("Data",fontsize=16)
            axes[0].set_ylabel('$\Delta$RA (arcsec)',fontsize=14)
            axes[0].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
            #plt.colorbar(im, ax=axes[0], shrink=0.5)
            im.set_clim(vmin, vmax)

            im = axes[1].imshow(img, origin='lower', cmap='inferno',extent=extent)
            axes[1].set_title("MCMC Model",fontsize=16)
            #plt.colorbar(im, ax=axes[1], shrink=0.5)
            im.set_clim(vmin, vmax)
            axes[1].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)

            im = axes[2].imshow(target_image-img, origin='lower', cmap='inferno',extent=extent,vmax=np.max(img))
            axes[2].set_title("Residuals",fontsize=16)
            im.set_clim(vmin, vmax)
            axes[2].set_xlabel('$\Delta$Dec (arcsec)',fontsize=14)
            plt.tight_layout()
            cb = plt.colorbar(im, ax=axes, shrink=0.5,pad=0.01)
            cb.set_label('Arbitrary Flux Units',fontsize=14)#,rotation=270)
            cb.ax.tick_params(labelsize=12)
            plt.savefig('../GPI_results/{}_MCMC_fit_{}knots.png'.format(name,num_knots))
            plt.show()

            labels = fit_keys[0:-1]
            for i in range(0, opt.spf_params['num_knots']):
                labels.append('k'+str(i+1))
                mc_model.plot_chains(labels)

            mc_model.show_corner_plot(labels, truths=mc_soln)
        return mc_soln