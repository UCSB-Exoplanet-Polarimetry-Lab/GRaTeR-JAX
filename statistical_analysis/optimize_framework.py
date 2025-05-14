import jax
import jax.numpy as jnp
from utils.SLD_utils import *
from scipy.optimize import minimize
from utils.mcmc_model import MCMC_model

# Built for old objective functions
class Optimizer:
    def __init__(self, DistrModel, FuncModel, PSF_Model, model_func, likelihood_func, pxInArcsec, distance, **kwargs):

        self.DistrModel = DistrModel
        self.FuncModel = FuncModel
        self.PSF_Model = PSF_Model
        self.model_func = model_func
        self.likelihood_func = likelihood_func
        self.pxInArcsec = pxInArcsec
        self.distance = distance
        self.add_params = kwargs

    def model(self, disk_params, spf_params):
        return self.model_func(self.DistrModel, self.FuncModel, self.PSF_Model, disk_params, spf_params,
                               pxInArcsec = self.pxInArcsec, distance = self.distance, **self.add_params)

    def log_likelihood_pos(self, disk_params, spf_params, target_image, err_map):
        return -self.likelihood_func(self.model(disk_params, spf_params), target_image, err_map)
    
    def log_likelihood(self, disk_params, spf_params, target_image, err_map):
        return self.likelihood_func(self.model(disk_params, spf_params), target_image, err_map)
    
    def scipy_optimize(self, init_disk_params, init_spf_params, target_image, err_map, disp_opt = False, disp_soln = False, iters = 500, grad = False, **kwargs):
        init_guess = jnp.concatenate([init_disk_params, init_spf_params])
        len_disk_params = jnp.size(init_disk_params)
        # 0: alpha_in, 1: alpha_out, 2: sma, 3: inclination, 4: position_angle, 5: xc, 6: yc, 7: e, 8: omega, 9: onwards is spline parameters
        llp = lambda x: self.log_likelihood_pos(x[0:len_disk_params], x[len_disk_params:], target_image, err_map)

        opt = {'disp':disp_opt,'maxiter':iters}
        soln = minimize(llp, init_guess, options=opt, **kwargs)
        if(disp_soln):
            print(soln)
        return soln.x[0:len_disk_params], soln.x[len_disk_params:]
    
    def mcmc(self, disk_params, spf_params, target_image, err_map, BOUNDS, nwalkers = 250, niter = 250, burns = 50):
        
        pars = np.concatenate([disk_params, jnp.log(spf_params)])

        if not(np.all(pars > BOUNDS[0]) and np.all(pars < BOUNDS[1])):
            print(pars[(pars < BOUNDS[0]) | (pars > BOUNDS[1])])
            print("Initial Parameters OUT OF BOUNDS!")
            return None

        len_disk_params = jnp.size(disk_params)
        llp = lambda x: self.log_likelihood(x[0:len_disk_params], jnp.exp(x[len_disk_params:]), target_image, err_map)

        mc_model = MCMC_model(llp, BOUNDS)
        mc_model.run(pars, nconst = 1e-7, nwalkers=nwalkers, niter=niter, burn_iter=burns)

        #mc_soln = np.median(mc_model.sampler.flatchain, axis=0) can get solution like this
        return mc_model


class OptimizeUtils:

    @classmethod
    def get_scaled_knots(cls, optimizer, knots, radius, inclination, position_angle, target_image,
                         init_SPF_func = DoubleHenyeyGreenstein_SPF, init_spf_params = [0.5, 0.5, 0.5]):
        ## Get a good scaling
        y, x = np.indices(target_image.shape)
        y -= 70
        x -= 70 
        rads = np.sqrt(x**2+y**2)
        mask = (rads > 12)

        init_knot_guess = init_SPF_func.compute_phase_function_from_cosphi(init_spf_params, optimizer.add_params['knots'])
        init_disk_guess = jnp.array([5., -5., radius, inclination, position_angle])
        init_cent_guess = jnp.array([70., 70.])

        init_image = optimizer.model(np.concatenate([init_disk_guess, init_cent_guess]), init_knot_guess)

        if inclination > 70: 
            knot_scale = 1.*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
        else: 
            knot_scale = 0.2*np.nanpercentile(target_image[mask], 99) / jnp.nanmax(init_image)
            
        init_knot_guess = DoubleHenyeyGreenstein_SPF.compute_phase_function_from_cosphi([0.5, 0.5, 0.5], knots) * knot_scale

        return init_knot_guess
    
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
    def process_image(cls, image, scale_factor=1, offset=1):
        cls.scaled_image = (image[::scale_factor, ::scale_factor])[1::, 1::]
        cropped_image = image[70:210, 70:210]
        def safe_float32_conversion(value):
            try:
                return np.float32(value)
            except (ValueError, TypeError):
                print("This value is unjaxable: " + str(value))
        fin_image = np.nan_to_num(cropped_image)
        fin_image = np.vectorize(safe_float32_conversion)(fin_image)
        return fin_image

    @classmethod
    def get_inc_bounded_knots(cls, inclination, radius, buffer = 0, num_knots=-1):
        if(num_knots <= 0):
            if(radius < 50):
                num_knots = 4
            else:
                num_knots = 6
        return jnp.linspace(jnp.cos(jnp.deg2rad(90-inclination-buffer)), jnp.cos(jnp.deg2rad(90+inclination+buffer)), num_knots)
    
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