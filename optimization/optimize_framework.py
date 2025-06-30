import jax
import jax.numpy as jnp
from disk_model.SLD_utils import *
from scipy.optimize import minimize
from optimization.mcmc_model import MCMC_model
from disk_model.objective_functions import objective_model, objective_ll, objective_fit, log_likelihood
import json

# Built for new objective function
class Optimizer:
    def __init__(self, disk_params, spf_params, psf_params, misc_params, DiskModel, DistrModel, FuncModel,
                 PSFModel, StellarPSFModel = None, stellar_psf_params = None, **kwargs):
        self.disk_params = disk_params
        self.spf_params = spf_params
        self.psf_params = psf_params
        self.stellar_psf_params = stellar_psf_params
        self.misc_params = misc_params
        self.DiskModel = DiskModel
        self.DistrModel = DistrModel
        self.FuncModel = FuncModel
        self.PSFModel = PSFModel
        self.StellarPSFModel = StellarPSFModel
        self.kwargs = kwargs
        self.name = 'test'
        self.last_fit = None

    def get_model(self):
        return objective_model(
            self.disk_params, self.spf_params, self.psf_params, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel,
            stellar_psf_params=self.stellar_psf_params, StellarPSFModel=self.StellarPSFModel,
            **self.kwargs
        )
    
    def get_disk(self):
        return objective_model(
            self.disk_params, self.spf_params, None, self.misc_params,
            self.DiskModel, self.DistrModel, self.FuncModel, None,
            stellar_psf_params=None, StellarPSFModel=None,
            **self.kwargs
        )

    def log_likelihood_pos(self, target_image, err_map):
        return -log_likelihood(self.get_model(), target_image, err_map)

    def log_likelihood(self, target_image, err_map):
        return log_likelihood(self.get_model(), target_image, err_map)
    
    def define_reference_images(self, reference_images):
        StellarPSFReference.reference_images = reference_images

    def scipy_optimize(self, fit_keys, logscaled_params, array_params, target_image, err_map,
                       disp_soln=False, iters=500, method=None, ftol=1e-12, gtol=1e-12, eps=1e-8, **kwargs): 
        
        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        llp = lambda x: -objective_fit(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys, self.disk_params,
                                       self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params,
                                       self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, self.StellarPSFModel,
                                       target_image, err_map)
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        soln = minimize(llp, init_x, method=method, options={'disp': True, 'maxiter': iters, 'ftol': ftol, 'gtol': gtol, 'eps': eps})

        param_list = self._unflatten_params(soln.x, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        if disp_soln:
            print(soln)

        self.last_fit = 'scipyminimize'

        return soln
    
    def scipy_bounded_optimize(self, fit_keys, fit_bounds, logscaled_params, array_params, target_image, err_map,
                       disp_soln=False, iters=500, ftol=1e-12, gtol=1e-12, eps=1e-8, scale_for_shape = False, **kwargs):
        
        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        scale = jnp.size(target_image) if scale_for_shape else 1.
        
        llp = lambda x: -objective_fit(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys, self.disk_params,
                                       self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params,
                                       self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, self.StellarPSFModel,
                                       target_image, err_map, scale=scale)
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        lower_bounds, upper_bounds = fit_bounds
        bounds = []
        i = 0
        for key, low, high in zip(fit_keys, lower_bounds, upper_bounds):
            low = np.atleast_1d(low)
            high = np.atleast_1d(high)
            if is_arrays[i]:
                for l, h in zip(low, high):
                    if logscales[i]:
                        bounds.append((np.log(l+1e-14), np.log(h)))
                    else:
                        bounds.append((low[0], high[0]))
            else:
                if logscales[i]:
                    bounds.append((np.log(low+1e-14), np.log(high)))
                else:
                    bounds.append((low[0], high[0]))
            i+=1
        soln = minimize(llp, init_x, method='L-BFGS-B', bounds=bounds, options={'disp': True, 'maxiter': iters, 'ftol': ftol, 'gtol': gtol, 'eps': eps})

        param_list = self._unflatten_params(soln.x, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        if disp_soln:
            print(soln)

        self.last_fit = 'scipyboundminimize'

        return soln

    def mcmc(self, fit_keys, logscaled_params, array_params, target_image, err_map, BOUNDS, nwalkers=250, niter=250, burns=50, 
            continue_from=False, scale_for_shape=False):
        logscales = self._highlight_selected_params(fit_keys, logscaled_params)
        is_arrays = self._highlight_selected_params(fit_keys, array_params)

        scale = jnp.size(target_image) if scale_for_shape else 1.
        
        ll = lambda x: objective_fit(self._unflatten_params(x, fit_keys, logscales, is_arrays), fit_keys, self.disk_params,
                                     self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params,
                                     self.DiskModel, self.DistrModel, self.FuncModel, self.PSFModel, self.StellarPSFModel,
                                     target_image, err_map, scale=scale)
        
        init_x = self._flatten_params(fit_keys, logscales, is_arrays)

        lower_bounds, upper_bounds = BOUNDS
        i = 0
        bounds = []
        for key, low, high in zip(fit_keys, lower_bounds, upper_bounds):
            low = np.atleast_1d(low)
            high = np.atleast_1d(high)
            if is_arrays[i]:
                if logscales[i]:
                    for l, h in zip(low, high):
                        bounds.append((np.log(np.maximum(l, 1e-14)), np.log(h)))
                else:
                    for l, h in zip(low, high):
                        bounds.append((l, h))
            else:
                if logscales[i]:
                    bounds.append((np.log(np.maximum(low[0], 1e-14)), np.log(high[0])))
                else:
                    bounds.append((low[0], high[0]))
            i+=1

        # Flatten the bound arrays for comparison
        init_lb, init_ub = zip(*bounds)
        init_lb = np.array(init_lb)
        init_ub = np.array(init_ub)

        if not (np.all(init_x >= init_lb) and np.all(init_x <= init_ub)):
            init_param_list = self._unflatten_params(init_x, fit_keys, logscales, is_arrays)
            init_lb_list = self._unflatten_params(init_lb, fit_keys, logscales, is_arrays)
            init_ub_list = self._unflatten_params(init_ub, fit_keys, logscales, is_arrays)
            print("Initial mcmc parameters are out of bounds!")
            output_string = ""
            for i in range(0, len(init_param_list)):
                if(np.any(init_param_list[i] < init_lb_list[i]) or np.any(init_param_list[i] > init_ub_list[i])):
                    output_string += (f"{fit_keys[i]}: {init_param_list[i]}, ")
            print(output_string[0:-2])
            raise Exception("MCMC Initial Bounds Exception")

        mc_model = MCMC_model(ll, (init_lb, init_ub), self.name)
        mc_model.run(init_x, nconst=1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns,continue_from=continue_from)

        mc_soln = mc_model.get_theta_median()
        param_list = self._unflatten_params(mc_soln, fit_keys, logscales, is_arrays)
        self._update_params(param_list, fit_keys)

        self.last_fit = 'mcmc'

        # Unlogscale the internal sampler chain
        array_lengths = [len(self._get_param_value(k)) if k in array_params else 1 for k in fit_keys]
        OptimizeUtils.unlogscale_mcmc_model(mc_model, fit_keys, logscaled_params, array_params, array_lengths)

        return mc_model
    
    def inc_bound_knots(self, buffer = 0):
        if(self.spf_params['num_knots'] <= 0):
            if(self.disk_params['sma'] < 50):
                self.spf_params['num_knots'] = 4
            else:
                self.spf_params['num_knots'] = 6
        self.spf_params['forwardscatt_bound'] = jnp.cos(jnp.deg2rad(90-self.disk_params['inclination']-buffer))
        self.spf_params['backscatt_bound'] = jnp.cos(jnp.deg2rad(90+self.disk_params['inclination']+buffer))
        return self.spf_params
    
    def scale_initial_knots(self, target_image, dhg_params = [0.5, 0.5, 0.5]):
        ## Get a good scaling
        y, x = np.indices(target_image.shape)
        y -= 70
        x -= 70 
        rads = np.sqrt(x**2+y**2)
        mask = (rads > 12)

        self.spf_params['knot_values'] = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi(dhg_params, InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params))

        init_image = self.get_model()

        if self.disk_params['inclination'] > 70: 
            knot_scale = 1.*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
        else: 
            knot_scale = 0.2*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
            
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * knot_scale

        #if self.FuncModel == FixedInterpolatedUnivariateSpline_SPF:
            #adjust_scale = 1.0 / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
                #InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
                #0.0)
            #self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
            #self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale
        #else:
        self.scale_spline_to_fixed_point(0, 1)

    def computer_stellar_psf_image(self):
        return self.StellarPSFModel.compute_stellar_psf_image(self.StellarPSFModel.pack_pars(self.stellar_psf_params),
                                                              self.misc_params['nx'], self.misc_params['ny'])

    def scale_spline_to_fixed_point(self, cosphi, spline_val):
        adjust_scale = spline_val / InterpolatedUnivariateSpline_SPF.compute_phase_function_from_cosphi(
            InterpolatedUnivariateSpline_SPF.init(self.spf_params['knot_values'], InterpolatedUnivariateSpline_SPF.get_knots(self.spf_params)),
            cosphi)
        self.spf_params['knot_values'] = self.spf_params['knot_values'] * adjust_scale
        self.misc_params['flux_scaling'] = self.misc_params['flux_scaling'] / adjust_scale

    def fix_all_nonphysical_params(self):
        if issubclass(self.FuncModel, InterpolatedUnivariateSpline_SPF):
            self.spf_params['knot_values'] = np.where(self.spf_params['knot_values'] < 1e-8, 1e-8, self.spf_params['knot_values'])
        if self.disk_params['e']<0:
            self.disk_params['e'] = 0

    def print_params(self):
        print("Disk Params: " + str(self.disk_params))
        print("SPF Params: " + str(self.spf_params))
        print("PSF Params: " + str(self.psf_params))
        print("Stellar PSF Params: " + str(self.stellar_psf_params))
        print("Misc Params: " + str(self.misc_params))

    def _flatten_params(self, fit_keys, logscales, is_arrays):
        """
        Flatten parameters into a 1D array for optimization.
        
        Parameters:
        -----------
        fit_keys : list
            List of parameter keys to be included in the flattened array.
        logscales : list
            List of boolean values indicating whether each parameter should be log-scaled.
            Must be the same length as fit_keys.
        is_arrays : list
            List of boolean values indicating whether each parameter is an array.
            Must be the same length as fit_keys.
            
        Returns:
        --------
        numpy.ndarray
            Flattened parameter array.
        """
            
        # Ensure lists are the same length as fit_keys
        if len(logscales) != len(fit_keys) or len(is_arrays) != len(fit_keys):
            raise ValueError("scales and is_arrays must have the same length as fit_keys")
        
        param_list = []
        for i, key in enumerate(fit_keys):
            # Get parameter from appropriate dictionary
            if isinstance(self.disk_params, dict) and key in self.disk_params:
                value = self.disk_params[key]
            elif isinstance(self.spf_params, dict) and key in self.spf_params:
                value = self.spf_params[key]
            elif isinstance(self.psf_params, dict) and key in self.psf_params:
                value = self.psf_params[key]
            elif isinstance(self.stellar_psf_params, dict) and key in self.stellar_psf_params:
                value = self.stellar_psf_params[key]
            elif isinstance(self.misc_params, dict) and key in self.misc_params:
                value = self.misc_params[key]
            else:
                raise ValueError(f"{key} not in any of the parameter dictionaries!")
                
            # Apply log scaling if needed
            if logscales[i]:
                if is_arrays[i]:
                    # Handle array parameters with log scaling
                    value = np.log(np.maximum(value, 1e-14))  # Ensure positive values for log
                else:
                    # Handle scalar parameters with log scaling
                    value = np.log(max(value, 1e-14))
                    
            param_list.append(value)
                
        return np.concatenate([np.atleast_1d(x) for x in param_list])

    def _unflatten_params(self, flattened_params, fit_keys, logscales, is_arrays):
        """
        Convert flattened parameter array back to appropriate parameter values.
        
        Parameters:
        -----------
        flattened_params : numpy.ndarray
            Flattened parameter array.
        fit_keys : list
            List of parameter keys corresponding to the flattened parameters.
        logscales : list
            List of boolean values indicating whether each parameter should be log-scaled.
            Must be the same length as fit_keys.
        is_arrays : list
            List of boolean values indicating whether each parameter is an array.
            Must be the same length as fit_keys.
            
        Returns:
        --------
        list
            List of unflattened parameter values.
        """
            
        # Ensure lists are the same length as fit_keys
        if len(logscales) != len(fit_keys) or len(is_arrays) != len(fit_keys):
            raise ValueError("scales and is_arrays must have the same length as fit_keys")
        
        param_list = []
        index = 0
        
        for i, key in enumerate(fit_keys):
            if is_arrays[i]:
                # For arrays, determine the size
                for param_dict in [self.disk_params, self.spf_params, self.psf_params, self.stellar_psf_params, self.misc_params]:
                    if isinstance(param_dict, dict) and key in param_dict and hasattr(param_dict[key], "__len__"):
                        array_size = len(param_dict[key])
                        break
                else:
                    raise ValueError(f"Cannot determine array size for {key}")
                
                # Extract array values
                array_values = flattened_params[index:index+array_size]
                index += array_size
                
                # Apply inverse scaling if needed
                if logscales[i]:
                    array_values = np.exp(array_values)
                    
                param_list.append(array_values)
            else:
                # Handle scalar parameters
                value = flattened_params[index]
                index += 1
                
                # Apply inverse scaling if needed
                if logscales[i]:
                    value = np.exp(value)
                    
                param_list.append(value)
                
        return param_list

    def _update_params(self, param_values, fit_keys):
        """
        Update class parameter dictionaries with new values.
        
        Parameters:
        -----------
        param_values : list
            List of parameter values.
        fit_keys : list
            List of parameter keys corresponding to the parameter values.
        """
        if len(param_values) != len(fit_keys):
            raise ValueError("param_values must have the same length as fit_keys")
            
        for i, key in enumerate(fit_keys):
            value = param_values[i]
            
            if isinstance(self.disk_params, dict) and key in self.disk_params:
                self.disk_params[key] = value
            elif isinstance(self.spf_params, dict) and key in self.spf_params:
                self.spf_params[key] = value
            elif isinstance(self.psf_params, dict) and key in self.psf_params:
                self.psf_params[key] = value
            elif isinstance(self.stellar_psf_params, dict) and key in self.stellar_psf_params:
                self.stellar_psf_params[key] = value
            elif isinstance(self.misc_params, dict) and key in self.misc_params:
                self.misc_params[key] = value
            else:
                raise ValueError(f"{key} not in any of the parameter dictionaries!")
        
        self.fix_all_nonphysical_params()

    def _highlight_selected_params(self, fit_keys, selected_params):
        select_bools = []
        for key in fit_keys:
            select_bools.append(key in selected_params)
        return select_bools

    def save_human_readable(self,dirname):
        with open(os.path.join(dirname,'{}_{}_hrparams.txt'.format(self.name,self.last_fit)), 'w') as save_file:
            save_file.write('Model Name: {}\n \n'.format(self.name))
            save_file.write('Method: {}\n \n'.format(self.last_fit))
            save_file.write('### Disk Params ### \n')
            for key in self.disk_params:
                save_file.write("{}: {}\n".format(key, self.disk_params[key]))
            save_file.write('\n### SPF Params ### \n')
            for key in self.spf_params:
                save_file.write("{}: {}\n".format(key, self.spf_params[key]))
            save_file.write('\n### PSF Params ### \n')
            for key in self.psf_params:
                save_file.write("{}: {}\n".format(key, self.psf_params[key]))
            save_file.write('\n### Misc Params ### \n')
            for key in self.misc_params:
                save_file.write("{}: {}\n".format(key, self.misc_params[key]))
        print("Saved human readable file to {}".format(os.path.join(dirname,'{}_{}_hrparams.txt'.format(self.name,self.last_fit))))

    def _get_param_value(self, key):
        param_dicts = [self.disk_params, self.spf_params, self.stellar_psf_params, self.misc_params]
        if isinstance(self.psf_params, dict):
            param_dicts.append(self.psf_params)
        for param_dict in param_dicts:
            if key in param_dict:
                return param_dict[key]
        raise KeyError(f"{key} not found in any parameter dict.")

    def save_machine_readable(self,dirname):
        with open(os.path.join(dirname,'{}_{}_diskparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            json.dump(self.disk_params, save_file)
        with open(os.path.join(dirname,'{}_{}_spfparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            serializable_spf = {}
            for key, value in self.spf_params.items():
                if isinstance(value, jnp.ndarray):
                    serializable_spf[key] = value.tolist()
                else:
                    serializable_spf[key] = value
            json.dump(serializable_spf, save_file)
        with open(os.path.join(dirname,'{}_{}_psfparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            json.dump(self.psf_params, save_file)
        with open(os.path.join(dirname,'{}_{}_miscparams.json'.format(self.name,self.last_fit)), 'w') as save_file:
            serializable_misc = {}
            for key, value in self.misc_params.items():
                if isinstance(value, jnp.ndarray):
                    serializable_misc[key] = value.tolist()
                else:
                    serializable_misc[key] = value
            json.dump(serializable_misc, save_file)
        print("Saved machine readable files to json in "+dirname)
    
    def load_machine_readable(self,dirname,method=None):
        ### defaults to last fitting mechanism, but can be changed to scipyminimize, scipyboundminimize, or mcmc
        if method == None:
            method = self.last_fit
        if self.last_fit == None:
            raise Exception("No last fit to load from. Please run a fit before loading.")
        else:
            try:
                with open(os.path.join(dirname,'{}_{}_diskparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    self.disk_params = json.load(read_file)
                with open(os.path.join(dirname,'{}_{}_spfparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    serializable_spf = json.load(read_file)
                    for key, value in serializable_spf.items():
                        if isinstance(value, list):
                            self.spf_params[key] = jnp.array(value)
                        else:
                            self.spf_params[key] = value
                with open(os.path.join(dirname,'{}_{}_psfparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    self.psf_params = json.load(read_file)
                with open(os.path.join(dirname,'{}_{}_miscparams.json'.format(self.name,self.last_fit)), 'r') as read_file:
                    serializable_misc = json.load(read_file)
                    for key, value in serializable_misc.items():
                        if isinstance(value, list):
                            self.misc_params[key] = jnp.array(value)
                        else:
                            self.misc_params[key] = value
                print("Loaded machine readable files from json in "+dirname)
            except FileNotFoundError:
                print("File not found. Please check the directory and file names.")
                return

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
    def convert_dhg_params_to_spline_params(cls, g1, g2, w, spf_params):
        return DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([g1, g2, w], InterpolatedUnivariateSpline_SPF.get_knots(spf_params))

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
    
    @classmethod
    def unlogscale_mcmc_model(cls, mc_model, fit_keys, logscaled_params, array_params, array_lengths):
        flat = mc_model.sampler.flatchain.copy()
        chain = mc_model.sampler.chain.copy()

        index = 0
        for i in range(len(fit_keys)):
            is_log = fit_keys[i] in logscaled_params
            is_array = fit_keys[i] in array_params
            length = array_lengths[i] if is_array else 1

            if is_log:
                flat[:, index:index+length] = np.exp(flat[:, index:index+length])
                chain[:, :, index:index+length] = np.exp(chain[:, :, index:index+length])
            index += length

        # Overwrite the sampler internals
        mc_model.sampler._chain = chain
        mc_model.sampler._flatchain = flat