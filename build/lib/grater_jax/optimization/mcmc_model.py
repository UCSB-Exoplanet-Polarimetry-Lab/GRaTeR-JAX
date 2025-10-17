"""
mcmc_model.py
=============

MCMC utilities for model fitting.

This module defines the `MCMC_model` class, a wrapper around the `emcee`
ensemble sampler with additional tools for running MCMC inference, analyzing
posterior distributions, and visualizing results. It provides built-in support
for HDF5 backends, custom priors, autocorrelation analysis, and plotting.

Main features
-------------
- `MCMC_model.run` : Run burn-in and production MCMC chains with emcee.
- `MCMC_model.get_theta_median`, `get_theta_percs`, `get_theta_max` :
  Extract parameter estimates from posterior samples.
- `MCMC_model.show_corner_plot` : Generate corner plots of posterior samples.
- `MCMC_model.plot_chains` : Visualize walker chains for diagnostics.
- `MCMC_model.auto_corr` : Estimate autocorrelation times.
"""

import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

class MCMC_model():
    """
    Markov Chain Monte Carlo (MCMC) wrapper for emcee sampler with tools for
    inference, diagnostics, and visualization.

    Parameters
    ----------
    fun : callable
        The log-likelihood function to evaluate.
    theta_bounds : tuple of np.ndarray
        Tuple (lower_bounds, upper_bounds) for parameters.
    name : str
        Name of the model (used for backend filename).
    """

    def __init__(self, fun, theta_bounds, name):
        self.fun = fun
        self.theta_bounds = theta_bounds
        self.sampler = None
        self.pos = None
        self.prob = None
        self.state = None
        self.name = name
        self.discard = 0
        self.nwalkers = None
        self.scaled_chain = None

    def _lnprior(self, theta):
        """
        Uniform prior within bounds.

        Parameters
        ----------
        theta : np.ndarray
            Parameter array.

        Returns
        -------
        float
            0 if within bounds, -inf otherwise.
        """
        if np.all(theta > self.theta_bounds[0]) and np.all(theta < self.theta_bounds[1]):
            return 0
        else:
            return -np.inf

    def _lnprob(self, theta, prior_func=None):
        """
        Log-probability function combining prior and likelihood.

        Parameters
        ----------
        theta : np.ndarray
            Parameter array.
        prior_func : callable, optional
            Custom prior function.

        Returns
        -------
        float
            Log-probability.
        """
        lp = prior_func(self.theta_bounds, theta) if prior_func else self._lnprior(theta)
        if lp == -np.inf:
            return -np.inf
        return lp + self.fun(theta)

    def run(self, initial, nwalkers=500, niter=500, burn_iter=100, nconst=1e-7, continue_from=None, **kwargs):
        """
        Run MCMC sampling using emcee.

        Parameters
        ----------
        initial : np.ndarray
            Initial guess for parameters.
        nwalkers : int
            Number of MCMC walkers.
        niter : int
            Number of production iterations.
        burn_iter : int
            Number of burn-in iterations.
        nconst : float
            Perturbation constant for initializing walkers.
        continue_from : bool or None
            Whether to continue from previous run.

        Returns
        -------
        tuple
            (sampler, chain, log-probs, random state)
        """
        self.ndim = len(initial)
        self.nwalkers = nwalkers
        self.niter = niter
        self.discard = burn_iter

        outfile = f"{self.name}_emcee_backend.h5"
        backend = emcee.backends.HDFBackend(outfile)

        if continue_from is not True:
            yes = input("This is going to overwrite the previous backend. Do you want to continue? (y/n): ")
            if yes == 'y':
                backend.reset(nwalkers, self.ndim)
                p0 = [np.array(initial) + nconst * np.random.randn(self.ndim) for _ in range(nwalkers)]
            else:
                print("Exiting...")
                return

            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob, backend=backend, **kwargs)
            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, burn_iter, progress=True)
            print("Running production...")
            sampler.run_mcmc(p0, niter, progress=True)

        else:
            sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._lnprob, backend=backend, **kwargs)
            sampler.run_mcmc(None, niter, progress=True)

        self.sampler = sampler
        self.full_chain = sampler.get_chain()
        self.full_lnprob = sampler.get_log_prob()
        self.pos = self.full_chain[-1]
        self.prob = self.full_lnprob[-1]
        self.state = sampler.random_state

        return sampler, self.full_chain, self.full_lnprob, self.state

    def set_discarded_iters(self, new_discard_iters):
        """
        Set the number of burn-in iterations to discard.

        Parameters
        ----------
        new_discard_iters : int
            Number of iterations to discard (must be >= 0).
        """
        if isinstance(new_discard_iters, int) and new_discard_iters >= 0:
            self.discard = new_discard_iters
        else:
            raise ValueError("Input valid discard value (int>=0)")

    def get_theta_median(self, scaled=False):
        """
        Return median of posterior samples.

        Parameters
        ----------
        scaled : bool, optional
            If True, use scaled parameter samples.

        Returns
        -------
        np.ndarray
            Median parameter values.
        """
        flatchain = self._get_flatchain(scaled)
        return np.median(flatchain, axis=0)

    def get_theta_percs(self, scaled=False):
        """
        Return 16th, 50th, and 84th percentiles of posterior samples.

        Parameters
        ----------
        scaled : bool, optional
            If True, use scaled parameter samples.

        Returns
        -------
        np.ndarray
            Array of shape (3, ndim) with 16th, 50th, 84th percentiles.
        """
        flatchain = self._get_flatchain(scaled)
        return np.percentile(flatchain, [16, 50, 84], axis=0)

    def get_theta_max(self, scaled=False):
        """
        Return parameter values corresponding to maximum log-probability.

        Parameters
        ----------
        scaled : bool, optional
            If True, return scaled values.

        Returns
        -------
        np.ndarray
            Parameter values with maximum log probability.
        """
        flatlnprob = self._get_flatlogprob()
        max_idx = np.argmax(flatlnprob)
        return self._get_flatchain(scaled)[max_idx]

    def show_corner_plot(self, labels, truths=None, show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84], quiet=False, scaled=False):
        """
        Generate corner plot using posterior samples.

        Parameters
        ----------
        labels : list of str
            List of parameter names for plot axes.
        truths : array-like, optional
            True values to show as vertical lines.
        show_titles : bool, optional
            If True, show titles with stats above plots.
        plot_datapoints : bool, optional
            If True, show individual data points.
        quantiles : list of float, optional
            Quantiles to show on plot.
        quiet : bool, optional
            Suppress text output.
        scaled : bool, optional
            If True, use scaled parameter samples.

        Returns
        -------
        matplotlib.figure.Figure
            The corner plot figure.
        """
        flatchain = self._get_flatchain(scaled)
        return corner.corner(flatchain, truths=truths, show_titles=show_titles, labels=labels, plot_datapoints=plot_datapoints, quantiles=quantiles, quiet=quiet)

    def plot_chains(self, labels, cols_per_row=3, scaled=False):
        """
        Plot individual walker chains for each parameter.

        Parameters
        ----------
        labels : list of str
            Names of parameters.
        cols_per_row : int, optional
            Number of subplot columns per row.
        scaled : bool, optional
            If True, use scaled parameter samples.
        """
        chain = self._get_chain(scaled)
        niter, nwalkers, n_params = chain.shape
        n_rows = int(np.ceil(n_params / cols_per_row))
        fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(6 * cols_per_row, 4 * n_rows), squeeze=False)
        fig.subplots_adjust(hspace=0.4)
        x = np.arange(niter)
        for idx in range(n_params):
            i, j = divmod(idx, cols_per_row)
            for walker in range(nwalkers):
                axes[i][j].plot(x, chain[:, walker, idx], alpha=0.5)
            axes[i][j].set_title(labels[idx])
        plt.show()

    def auto_corr(self, chain_length=50):
        """
        Estimate autocorrelation length using Sokal's method.

        Parameters
        ----------
        chain_length : int
            Number of subchain lengths to evaluate.

        Returns
        -------
        np.ndarray
            Autocorrelation estimates for different subchain lengths.
        """
        chain = self.sampler.get_chain()[:, :, 0].T
        N = np.exp(np.linspace(np.log(100), np.log(chain.shape[1]), chain_length)).astype(int)
        estims = np.empty(len(N))
        for i, n in enumerate(N):
            estims[i] = self._autocorr_new(chain[:, :n])
        return estims

    def _next_pow_two(self, n):
        """
        Return next power of 2 greater than or equal to `n`.

        Parameters
        ----------
        n : int
            Input integer.

        Returns
        -------
        int
            Smallest power of 2 >= n.
        """
        if n < 1:
            return 1
        return 1 << (n - 1).bit_length()

    def _autocorr_func_1d(self, x):
        """
        Compute unnormalized autocorrelation function using FFT.

        Parameters
        ----------
        x : np.ndarray
            1D input signal.

        Returns
        -------
        np.ndarray
            Autocorrelation function.
        """
        x = np.atleast_1d(x)
        n = self._next_pow_two(len(x))
        f = np.fft.fft(x - np.mean(x), n=2 * n)
        acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
        return acf / (4 * n)

    def _auto_window(self, taus, c):
        """
        Automated window size selection for autocorrelation truncation.

        Parameters
        ----------
        taus : np.ndarray
            Autocorrelation time estimates.
        c : float
            Threshold multiplier.

        Returns
        -------
        int
            Truncation window index.
        """
        m = np.arange(len(taus)) < c * taus
        return np.argmin(m) if np.any(m) else len(taus) - 1

    def _autocorr_new(self, y, c=5.0):
        """
        Compute autocorrelation estimator for 2D array.

        Parameters
        ----------
        y : np.ndarray
            2D array of shape (n_walkers, n_steps).
        c : float, optional
            Window scaling constant.

        Returns
        -------
        float
            Final autocorrelation estimate.
        """
        f = np.zeros(y.shape[1])
        for yy in y:
            f += self._autocorr_func_1d(yy)
        f /= len(y)
        taus = 2.0 * np.cumsum(f) - 1.0
        window = self._auto_window(taus, c)
        return taus[window]

    def _get_chain(self, scaled=False):
        """
        Return MCMC chain after discarding burn-in.

        Parameters
        ----------
        scaled : bool, optional
            If True, return scaled samples.

        Returns
        -------
        np.ndarray
            Chain of shape (niter, nwalkers, ndim).
        """
        discard = min(self.discard, self.sampler.get_chain().shape[0])
        return self.scaled_chain[discard:, :, :] if scaled else self.sampler.get_chain()[discard:, :, :]

    def _get_flatchain(self, scaled=False):
        """
        Return flattened MCMC chain.

        Parameters
        ----------
        scaled : bool, optional
            If True, use scaled samples.

        Returns
        -------
        np.ndarray
            Flattened chain of shape (niter * nwalkers, ndim).
        """
        return self._get_chain(scaled).reshape((-1, self.ndim))

    def _get_flatlogprob(self):
        """
        Return flattened log-probability values after burn-in.

        Returns
        -------
        np.ndarray
            Flattened log probability array.
        """
        discard = min(self.discard, self.sampler.get_chain().shape[0])
        return self.sampler.get_log_prob(discard=discard, flat=True)
