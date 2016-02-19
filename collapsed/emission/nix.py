import copy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln

class DiagNormal(object):
    def __init__(self, mu, sigma2):
        self.mu = mu
        self.inv_two_sigma2 = 1.0 / (2 * sigma2)
        self.logZ = np.log(np.sqrt(2 * np.pi * sigma2)).sum()

    def logpdf(self, x):
        diff = x - self.mu
        lp = -(diff* diff * self.inv_two_sigma2).sum(-1)
        return lp - self.logZ

class NIXUnit(object):
    """Compound Normal-Inverse-Chi2.
    """
    def __init__(self, n_dims, kappa_0, mu_0, nu_0, sigma2_0, min_covar_diag=0.01):
        """
        Parameters
        ----------
        kappa_0: float
            relative precision hyperparameter
        mu_0: np.ndarray
            prior mean
        nu_0: float
            degrees of freedom
        sigma2_0: np.ndarray
            prior variance
        """

        self.n_dims = n_dims
        self.min_covar_diag = min_covar_diag
        
        self.kappa_0 = float(kappa_0)
        self.mu_0 = np.array(mu_0).astype(float)

        self.nu_0 = float(nu_0)
        self.sigma2_0 = np.array(sigma2_0).astype(float)

        if self.mu_0.shape != (n_dims,):
            raise ValueError('mu_0 has wrong shape. given: {0}. expected: {1}.'.format(self.mu_0.shape, (n_dims,)))

        if self.sigma2_0.shape != (n_dims,):
            raise ValueError('sigma2_0 has wrong shape. given: {0}. expected: {1}.'.format(self.sigma2_0.shape, (n_dims,)))
        
        if nu_0 <= 2:
            raise ValueError('nu_0 must be greater than 2. given: {0}'.format(nu_0))

        self.reset()

    def reset(self):
        self.x_sum = np.zeros_like(self.mu_0) # sum_i x_i
        self.s_sum = np.zeros_like(self.sigma2_0) # sum_i [x_{i,1} * x_{i,1},...,x_{1,d} * x_{1,d}]
        self.n = 0.0

        self.x_total = np.zeros_like(self.mu_0)
        self.s_total = np.zeros_like(self.sigma2_0)
        self.n_total = 0.0
        
    def posterior(self, weight=1.0):
        n = float(self.n)
        d = self.n_dims
        
        kappa_0 = self.kappa_0
        mu_0 = self.mu_0
        nu_0 = self.nu_0
        sigma2_0 = self.sigma2_0
        x_sum = self.x_sum
        s_sum = self.s_sum

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n

        x_bar = x_sum / n if n > 0 else np.zeros_like(x_sum)
        ss_diff = s_sum - n * x_bar * x_bar
        ms_diff = ((n * kappa_0) / (kappa_n)) * (x_bar - mu_0)**2

        mu_n = (kappa_0 * mu_0 + x_sum) / kappa_n
        sigma2_n = (nu_0 * sigma2_0 + ss_diff + ms_diff) / nu_n

        return NIXUnit(self.n_dims, weight * kappa_n, mu_n, weight * nu_n, sigma2_n, self.min_covar_diag)

    def observe(self, x):
        self.x_sum += x
        self.s_sum += x * x
        self.n += 1

    def forget(self, x):
        self.x_sum -= x
        self.s_sum -= x * x
        self.n -= 1

    def accumulate(self, x):
        self.x_total += x
        self.s_total += x * x
        self.n_total += 1        

    def params(self):
        n = float(self.n)
        
        kappa_0 = self.kappa_0
        mu_0 = self.mu_0
        nu_0 = self.nu_0
        sigma2_0 = self.sigma2_0
        x_sum = self.x_sum
        s_sum = self.s_sum

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n
        x_bar = x_sum / n if n > 0 else np.zeros_like(x_sum)
        ss_diff = s_sum - n * x_bar * x_bar
        ms_diff = ((n * kappa_0) / (kappa_n)) * (x_bar - mu_0)**2
        
        mu = (kappa_0 * mu_0 + x_sum) / kappa_n
        sigma2 =  (nu_0 * sigma2_0 + ss_diff + ms_diff) / (nu_n - 2)
        sigma2 = np.maximum(sigma2, self.min_covar_diag)
        return mu, sigma2

    def expected_params(self):
        n = float(self.n_total)
        
        kappa_0 = self.kappa_0
        mu_0 = self.mu_0
        nu_0 = self.nu_0
        sigma2_0 = self.sigma2_0
        x_sum = self.x_total
        s_sum = self.s_total

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n
        x_bar = x_sum / n if n > 0 else np.zeros_like(x_sum)
        ss_diff = s_sum - n * x_bar * x_bar
        ms_diff = ((n * kappa_0) / (kappa_n)) * (x_bar - mu_0)**2
        
        mu = (kappa_0 * mu_0 + x_sum) / kappa_n
        sigma2 =  (nu_0 * sigma2_0 + ss_diff + ms_diff) / (nu_n - 2)
        sigma2 = np.maximum(sigma2, self.min_covar_diag)
        return mu, sigma2

    def generate_sample(self, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        mu, sigma = frozen_params
        return np.atleast_1d(multivariate_normal.rvs(mean=mu, cov=sigma))

    def marginal_likelihood(self, log=True, normalize=True):
        n = float(self.n)
        d = self.n_dims
        
        kappa_0 = self.kappa_0
        mu_0 = self.mu_0
        nu_0 = self.nu_0
        sigma2_0 = self.sigma2_0
        x_sum = self.x_sum
        s_sum = self.s_sum

        kappa_n = kappa_0 + n
        nu_n = nu_0 + n
        x_bar = x_sum / n if n > 0 else np.zeros_like(x_sum)
        ss_diff = s_sum - n * x_bar * x_bar
        ms_diff = ((n * kappa_0) / kappa_n) * (x_bar - mu_0)**2

        mu_n = (kappa_0 * mu_0 + x_sum) / kappa_n
        sigma2_n = (nu_0 * sigma2_0 + ss_diff + ms_diff) / nu_n

        loglike = 0.0
        loglike += d * (gammaln(0.5 * nu_n) - gammaln(0.5 * nu_0))
        loglike += d * 0.5 * (np.log(kappa_0) - np.log(kappa_n))
        loglike += (0.5 * nu_0 * (np.log(nu_0) + np.log(sigma2_0))).sum()
        loglike -= (0.5 * nu_n * (np.log(nu_n) + np.log(sigma2_n))).sum()
        loglike -= d * 0.5 * n * np.log(np.pi)

        return loglike if log else np.exp(loglike)

    def vector(self, obs, log=True, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        mu, sigma2 = frozen_params
        lpvec = DiagNormal(mu, sigma2).logpdf(obs)
        lpvec = np.maximum(lpvec, -700)
        return lpvec if log else np.exp(lpvec)


class NIX(object):
    def __init__(self, n_states, n_dims, kappa_0, mu_0, nu_0, sigma2_0, min_covar_diag=0.01):
        mu_0 = np.array(mu_0)
        sigma2_0 = np.array(sigma2_0)
        self.n_states = n_states
        if mu_0.ndim == 1:
            mu_0 = [mu_0.copy() for i in range(n_states)]
        if sigma2_0.ndim == 1:
            sigma2_0 = [sigma2_0.copy() for i in range(n_states)]

        self.components = [NIXUnit(n_dims, kappa_0, mu_0[i], nu_0, sigma2_0[i], min_covar_diag) for i in range(n_states)]
        self.kappa_0 = kappa_0
        self.mu_0 = mu_0
        self.nu_0 = nu_0
        self.sigma2_0 = sigma2_0
        self.min_covar_diag = min_covar_diag
        self.n_states = n_states
        self.n_dims = n_dims

    def reset(self):
        for component in self.components:
            component.reset()

    def posterior(self, weight):
        new = NIX(self.n_states, self.n_dims, self.kappa_0, self.mu_0, self.nu_0, self.sigma2_0, self.min_covar_diag)
        new.components = [component.posterior(weight) for component in self.components]
        return new

    def slice(self, i, j):
        new = NIX(j - i, self.n_dims, self.kappa_0, self.mu_0, self.nu_0, self.sigma2_0, self.min_covar_diag)
        new.components = [copy.deepcopy(component) for component in self.components[i:j]]
        return new

    def observe(self, states, obs):
        for k, x in zip(states, obs):
            self.components[k].observe(x)

    def forget(self, states, obs):
        for k, x in zip(states, obs):
            self.components[k].forget(x)

    def accumulate(self, states, obs):
        for k, x in zip(states, obs):
            self.components[k].accumulate(x)

    def params(self):
        return [component.params() for component in self.components]

    def expected_params(self):
        return [component.expected_params() for component in self.components]

    def marginal_likelihood(self, log=True, normalize=True):
        lp = sum(component.marginal_likelihood(log=True) for component in self.components)
        return lp if log else np.exp(lp)

    def generate_sample(self, state, frozen_params=None):
        if frozen_params is None:
            return self.components[state].generate_sample()
        if len(frozen_params) != self.n_states:
            raise ValueError('number of frozen params != number of components')
        return self.components[state].generate_sample(frozen_params=frozen_params[state])

    def matrix(self, obs, log=True, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        if len(frozen_params) != self.n_states:
            raise ValueError('number of frozen params != number of components')
        B = np.array([c.vector(obs, log=log, frozen_params=p) for c, p in zip(self.components, frozen_params)])
        return B.T.copy()
