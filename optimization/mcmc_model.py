import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

class MCMC_model():
    def __init__(self, fun, theta_bounds, name):
        self.fun = fun
        self.theta_bounds = theta_bounds
        self.sampler = None
        self.pos = None
        self.prob = None
        self.state = None
        self.name = name

    def _lnprior(self, theta):
        if np.all(theta > self.theta_bounds[0]) and np.all(theta < self.theta_bounds[1]):
            return 0
        else:
            return -np.inf

    def _lnprob(self, theta, prior_func = None):
        if prior_func == None:
            lp = self._lnprior(theta)
        else:
            lp = prior_func(self.theta_bounds, theta)
        if lp == -np.inf:
            return -np.inf
        return lp + self.fun(theta)

    def run(self, initial, nwalkers=500, niter = 500, burn_iter = 100, nconst = 1e-7, continue_from=None, **kwargs):

        self.ndim = len(initial)
        self.nwalkers = nwalkers
        self.niter = niter
        self.burn_iter = burn_iter

        outfile = "{}_emcee_backend.h5".format(self.name)
        backend = emcee.backends.HDFBackend(outfile)
        if continue_from is not True:
            yes = input("This is going to overwrite the previous backend. Do you want to continue? (y/n): ")
            if yes == 'y':
                print("Overwriting the previous backend...")
                backend.reset(nwalkers, self.ndim)
                p0 = [np.array(initial) + 1e-7 * np.random.randn(self.ndim) for i in range(nwalkers)]
            else:
                print("Exiting...")
                return
            
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob, backend=backend, **kwargs)
        if continue_from is not True:
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter,progress=True)
            sampler.reset()
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter,progress=True)
        elif continue_from is True:
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(None, niter,progress=True)  
        
        self.sampler, self.pos, self.prob, self.state = sampler, pos, prob, state
        return sampler, pos, prob, state

    def get_theta_max(self):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        return self.sampler.flatchain[np.argmax(self.sampler.flatlnprobability)]
    
    def get_theta_median(self):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        return np.median(self.sampler.flatchain, axis=0)

    def show_corner_plot(self, labels, discard=0, truths=None, show_titles=True, plot_datapoints=True, quantiles = [0.16, 0.5, 0.84],
                            quiet = False):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        fig = corner.corner(self.sampler.flatchain[:,discard:],truths=truths, show_titles=show_titles,labels=labels,
                                plot_datapoints=plot_datapoints,quantiles=quantiles, quiet=quiet)

    def plot_chains(self, labels, cols_per_row = 3):
        if self.sampler is None:
            raise Exception("Need to run model first!")
        
        chain = self.sampler.get_chain()  # shape: (niter, nwalkers, ndim)
        n_params = chain.shape[2]
        n_rows = int(np.ceil(n_params / cols_per_row))
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 4 * n_rows), squeeze=False)
        fig.subplots_adjust(hspace=0.4)

        x = np.arange(self.niter)
        for idx in range(n_params):
            i, j = divmod(idx, cols_per_row)
            for walker in range(self.nwalkers):
                axes[i][j].plot(x, chain[:, walker, idx], alpha=0.5)
            axes[i][j].set_title(labels[idx])
        plt.show()


    # Autocorrelation Methods from Here
    def auto_corr(self, chain_length = 50):
        if (self.sampler == None):
            raise Exception("Need to run model first!")
        chain = self.sampler.get_chain()[:, :, 0].T

        # Compute the estimators for a few different chain lengths
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), chain_length)).astype(int)
        estims = np.empty(len(N))
        for i, n in enumerate(N):
            estims[i] = self._autocorr_new(chain[:, :n])
        return estims

    def _next_pow_two(self, n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def _autocorr_func_1d(self, x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = self._next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        # Optionally normalize
        #if norm:
        #    acf /= acf[0]

        #return acf
    
    # Automated windowing procedure following Sokal (1989)
    def _auto_window(self, taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def _autocorr_new(self, y, c=5.0):
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self._autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self._auto_window(taus, c)
        return taus[window]
