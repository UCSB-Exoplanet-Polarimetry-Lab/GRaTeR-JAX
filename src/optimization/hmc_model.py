import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
import matplotlib.pyplot as plt
import corner
from random import random

class HMC_model:
    def __init__(self, fun, theta_bounds, rng_key=None):
        self.fun = fun
        self.theta_bounds = jnp.array(theta_bounds)
        self.rng_key = rng_key if rng_key else jax.random.PRNGKey(0)
        self.samples = None
        self.kernel = None
        self.mcmc = None

    def _model(self, theta=None):
        # Check if theta is within bounds
        in_bounds = jnp.all(theta >= self.theta_bounds[0]) & jnp.all(theta <= self.theta_bounds[1])
        logp = jax.lax.cond(in_bounds, lambda x: self.fun(x), lambda x: -jnp.inf, theta)
        return logp

    def run(self, initial, niter=500, nwarmup=100):
        init_params = {'theta': jnp.array(initial)}
        init_strategy = init_to_value(values=init_params)

        model_fun = lambda theta: -self._model(theta)

        self.kernel = NUTS(potential_fn=model_fun, init_strategy=init_strategy)
        self.mcmc = MCMC(self.kernel, num_warmup=nwarmup, num_samples=niter)
        #self.mcmc.run(self.rng_key, init_params=init_params)
        self.mcmc.run(self.rng_key, init_params=initial)
        
        self.samples = self.mcmc.get_samples()

    def get_theta_max(self):
        if self.samples is None:
            raise Exception("Need to run model first!")
        #log_probs = jax.vmap(self.fun)(self.samples['theta'])
        log_probs = jax.vmap(self.fun)(self.samples)
        #return self.samples['theta'][jnp.argmax(log_probs)]
        return self.samples[jnp.argmax(log_probs)]

    def get_theta_median(self):
        if self.samples is None:
            raise Exception("Need to run model first!")
        #return jnp.median(self.samples['theta'], axis=0)
        return jnp.median(self.samples, axis=0)

    def show_corner_plot(self, labels, truths=None, show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], quiet=False):
        if self.samples is None:
            raise Exception("Need to run model first!")
        
        # Convert samples to a numpy array
        #samples_array = np.array(self.samples['theta'])
        samples_array = np.array(self.samples)
        
        # Handle cases where some parameters might have no dynamic range
        ranges = [(np.min(samples_array[:, i]), np.max(samples_array[:, i])) for i in range(samples_array.shape[1])]
        
        # Add a small buffer to the ranges to avoid issues with zero dynamic range
        ranges = [(r[0] - 1e-4, r[1] + 1e-4) if r[0] == r[1] else r for r in ranges]
        
        # Create the corner plot with specified ranges
        fig = corner.corner(samples_array, truths=truths, show_titles=show_titles, labels=labels,
                            plot_datapoints=plot_datapoints, quantiles=quantiles, quiet=quiet, range=ranges)

    def plot_results(self, labels, figsize = (20,20)):
        if self.samples is None:
            raise Exception("Need to run model first!")

        #samples = self.samples['theta']
        samples = self.samples
        n_cols = int((samples.shape[1] + 2) / 3)
        fig, axes = plt.subplots(n_cols,3, figsize=figsize)
        axes = np.atleast_2d(axes)
        fig.subplots_adjust(hspace=0.5)
        for i in range(0, n_cols):
            for j in range(0, 3):
                if(3*i+j < samples.shape[1]):
                    axes[i][j].plot(np.linspace(1, samples.shape[0]+1, samples.shape[0]), samples[:, i])
                    axes[i][j].set_title(labels[3*i+j])

    def auto_corr(self, chain_length=50):
        if self.samples is None:
            raise Exception("Need to run model first!")
        #chain = self.mcmc.get_samples()['theta'][:, 0].T
        chain = self.mcmc.get_samples()[:, 0].T

        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), chain_length)).astype(int)
        estims = np.empty(len(N))
        for i, n in enumerate(N):
            estims[i] = self._autocorr_new(chain[:n])
        return estims

    def _next_pow_two(self, n):
        i = 1
        while i < n:
            i = i << 1
        return i

    def _autocorr_func_1d(self, x, norm=True):
        x = np.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("Invalid dimensions for 1D autocorrelation function")
        n = self._next_pow_two(len(x))

        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        acf /= 4 * n

        return acf

    def _auto_window(self, taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    def _autocorr_new(self, y, c=5.0):
        f = np.zeros(y.shape[0])
        for yy in y:
            f += self._autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self._auto_window(taus, c)
        return taus[window]
