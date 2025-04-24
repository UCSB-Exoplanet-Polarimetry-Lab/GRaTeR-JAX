import jax
import jax.numpy as jnp
from utils.SLD_utils import *
from scipy.optimize import minimize
from utils.mcmc_model import MCMC_model
from utils.objective_functions import objective_model, objective_ll, objective_fit, log_likelihood

# Built for new objective function
class Optimizer:
    def __init__(self, disk_params, spf_params, psf_params, misc_params, DiskModel, DistrModel, FuncModel, PSFModel, **kwargs):
        self.disk_params = disk_params
        self.spf_params = spf_params
        self.psf_params = psf_params
        self.misc_params = misc_params
        self.DiskModel = DiskModel
        self.DistrModel = DistrModel
        self.FuncModel = FuncModel
        self.PSFModel = PSFModel
        self.kwargs = kwargs

    def model(self):
        return objective_model(
            self.disk_params, self.spf_params, self.psf_params, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel,
            self.PSFModel, **self.kwargs
        )

    def log_likelihood_pos(self, target_image, err_map):
        return -log_likelihood(self.model(), target_image, err_map)

    def log_likelihood(self, target_image, err_map):
        return log_likelihood(self.model(), target_image, err_map)

    def scipy_optimize(self, fit_keys, target_image, err_map,
                       disp_opt=False, disp_soln=False, iters=500, grad=False, **kwargs):

        def expand(x):
            new_list = []
            index = 0
            for key in fit_keys:
                if key == "knot_values":
                    new_list.append(x[index:index+self.spf_params['num_knots']])
                    index+=self.spf_params['num_knots']
                else:
                    new_list.append(x[index])
                    index += 1
            return new_list

        llp = lambda x: -objective_fit(expand(x), fit_keys, self.disk_params, self.spf_params, self.psf_params, self.misc_params,
                                    self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, target_image, err_map)
        
        param_list = []
        for key in fit_keys:
            if key in self.disk_params:
                param_list.append(self.disk_params[key])
            elif key in self.spf_params:
                param_list.append(self.spf_params[key])
            elif key in self.psf_params:
                param_list.append(self.psf_params[key])
            elif key in self.misc_params:
                param_list.append(self.misc_params[key])
            else:
                print(key + " not in any of the parameter dictionaries!")
                fit_keys.pop(key)

        init_x = np.concatenate([np.atleast_1d(x) for x in param_list])
        soln = minimize(llp, init_x, options={'disp': True, 'max_itr': 500})

        params = 0
        param_list = expand(soln.x)
        for key in fit_keys:
            if key in self.disk_params:
                self.disk_params[key] = param_list[params]
            elif key in self.spf_params:
                self.spf_params[key] = param_list[params]
            elif key in self.psf_params:
                self.psf_params[key] = param_list[params]
            elif key in self.misc_params:
                self.misc_params[key] = param_list[params]
            else:
                print(key + " not in any of the parameter dictionaries!")
            params+=1

        opt = {'disp': disp_opt, 'maxiter': iters}
        if disp_soln:
            print(soln)
        return soln

    def mcmc(self, fit_keys, target_image, err_map, BOUNDS, nwalkers=250, niter=250, burns=50):

        def expand(x):
            new_list = []
            index = 0
            for key in fit_keys:
                if key == "knot_values":
                    new_list.append(np.exp(x[index:index+self.spf_params['num_knots']]))
                    index+=self.spf_params['num_knots']
                else:
                    new_list.append(x[index])
                    index += 1
            return new_list

        ll = lambda x: -objective_fit(expand(x), fit_keys, self.disk_params, self.spf_params, self.psf_params, self.misc_params,
                                    self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, target_image, err_map)
        
        param_list = []
        for key in fit_keys:
            if key in self.disk_params:
                param_list.append(self.disk_params[key])
            elif key in self.spf_params:
                if key == "knot_values":
                    param_list.append(np.log(self.spf_params[key]))
                else:
                    param_list.append(self.spf_params[key])
            elif key in self.psf_params:
                param_list.append(self.psf_params[key])
            elif key in self.misc_params:
                param_list.append(self.misc_params[key])
            else:
                print(key + " not in any of the parameter dictionaries!")
                fit_keys.pop(key)

        init_x = np.concatenate([np.atleast_1d(x) for x in param_list])
        init_lb = np.concatenate([np.atleast_1d(x) for x in BOUNDS[0]])
        init_ub = np.concatenate([np.atleast_1d(x) for x in BOUNDS[1]])
        if not (np.all(init_x > init_lb) and np.all(init_x < init_ub)):
            print("Initial parameters out of bounds:", init_x[(init_x < init_lb) | (init_x > init_ub)])
            return None

        mc_model = MCMC_model(ll, BOUNDS)
        mc_model.run(init_x, nconst=1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns)

        mc_soln = np.median(mc_model.sampler.flatchain, axis=0)
        params = 0
        param_list = expand(mc_soln)
        for key in fit_keys:
            if key in self.disk_params:
                self.disk_params[key] = param_list[params]
            elif key in self.spf_params:
                self.spf_params[key] = param_list[params]
            elif key in self.psf_params:
                self.psf_params[key] = param_list[params]
            elif key in self.misc_params:
                self.misc_params[key] = param_list[params]
            else:
                print(key + " not in any of the parameter dictionaries!")
            params+=1

        return mc_model
    
    def inc_bound_knots(self, buffer = 0):
        if(self.spf_params['num_knots'] <= 0):
            if(self.disk_params['sma'] < 50):
                self.spf_params['num_knots'] = 4
            else:
                self.spf_params['num_knots'] = 6
        self.spf_params['up_bound'] = jnp.cos(jnp.deg2rad(90-self.disk_params['inclination']-buffer))
        self.spf_params['low_bound'] = jnp.cos(jnp.deg2rad(90+self.disk_params['inclination']+buffer))
        return self.spf_params
    
    def scale_knots(self, target_image, dhg_params = [0.5, 0.5, 0.5]):
        ## Get a good scaling
        y, x = np.indices(target_image.shape)
        y -= 70
        x -= 70 
        rads = np.sqrt(x**2+y**2)
        mask = (rads > 12)

        self.spf_params['knot_values'] = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(dhg_params, InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params))

        init_image = self.model()

        if self.disk_params['inclination'] > 70: 
            knot_scale = 1.*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
        else: 
            knot_scale = 0.2*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
            
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * knot_scale

        if self.FuncModel == FixedInterpolatedUnivariateSpline_SPF:
            adjust_scale = 1.0 / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
                InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
                0.0)
            self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
            self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale

    def scale_spline_to_fixed_point(self, cosphi, spline_val):
        adjust_scale = spline_val / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
            cosphi)
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
        self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale

    def print_params(self):
        print("Disk Params: " + str(self.disk_params))
        print("SPF Params: " + str(self.spf_params))
        print("PSF Params: " + str(self.psf_params))
        print("Misc Params: " + str(self.misc_params))

class OptimizeUtils:
    
    @classmethod
    def create_empirical_err_map(cls, data, annulus_width=5, mask_rad=9, outlier_pixels=None):    
        y,x = np.indices(data.shape)
        y -= data.shape[0]//2
        x -= data.shape[1]//2 
        radii = np.sqrt(x**2 + y**2) 
        noise_array = np.zeros_like(data)
        for i in range(0, int(np.max(radii)//annulus_width) ): 
            indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
            noise_array[indices] = np.nanstd(data[indices])
        mask = radii <= mask_rad
        noise_array[mask] = 1e10

        if(outlier_pixels != None):
            for pixel in outlier_pixels:
                noise_array[pixel[0]][pixel[1]] = noise_array[pixel[0]][pixel[1]] * 1e6 

        return noise_array

    @classmethod
    def process_image(cls, image, scale_factor=1, bounds = (70, 210, 70, 210)):
        cls.scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
        cropped_image = image[bounds[0]:bounds[1],bounds[0]:bounds[1]]
        def safe_float32_conversion(value):
            try:
                return np.float32(value)
            except (ValueError, TypeError):
                print("This value is unjaxable: " + str(value))
        fin_image = np.nan_to_num(cropped_image)
        fin_image = np.vectorize(safe_float32_conversion)(fin_image)
        return fin_image
    
    @classmethod
    def get_mask(cls, data, annulus_width=5, mask_rad=9, outlier_pixels=None):    
        y,x = np.indices(data.shape)
        y -= data.shape[0]//2
        x -= data.shape[1]//2 
        radii = np.sqrt(x**2 + y**2) 
        noise_array = np.zeros_like(data)
        for i in range(0, int(np.max(radii)//annulus_width) ): 
            indices = (radii > i*annulus_width) & (radii <= (i+1)*annulus_width) 
            noise_array[indices] = np.nanstd(data[indices])
        mask = radii <= mask_rad

        return mask