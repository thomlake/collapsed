import copy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln


class NIWUnit(object):
    """Compound Normal-Inverse-Wishart.
    """
    def __init__(self, n_dims, k0, m0, v0, S0, min_covar_diag=0.01):
        """
        Parameters
        ----------
        n_dims: int
            input dimensionality
        k0: float
            relative precision hyperparameter
        m0: np.ndarray
            prior mean
        v0: float
            degrees of freedom
        S0: np.ndarray
            prior covariance
        """

        self.n_dims = n_dims
        self.min_covar_diag = min_covar_diag
        
        self.k0 = float(k0)
        self.m0 = np.array(m0).astype(float)

        self.v0 = float(max(v0, self.n_dims + 2.))
        self.S0 = np.array(S0).astype(float)

        self.reset()

    def reset(self):
        self.x = np.zeros_like(self.m0) # sum_i x_i
        self.S = np.zeros_like(self.S0) # sum_i x_i x_i^T
        self.n = 0.0

        self.x_total = np.zeros_like(self.m0)
        self.S_total = np.zeros_like(self.S0)
        self.n_total = 0.0
        
    def posterior(self, weight=1.0):
        n = float(self.n)
        d = self.n_dims
        
        k0 = self.k0
        m0 = self.m0
        v0 = self.v0
        S0 = self.S0

        kn = k0 + n
        mn = (k0 * m0 + self.x) / kn
        vn = v0 + n
        Sn = S0 + self.S + k0 * np.outer(m0, m0) - kn * np.outer(mn, mn)
        return NIWUnit(weight * kn, mn, weight * vn, Sn, self.min_covar_diag)

    def observe(self, x):
        self.x += x
        self.S += np.outer(x, x)
        self.n += 1

    def forget(self, x):
        self.x -= x
        self.S -= np.outer(x, x)
        self.n -= 1

    def accumulate(self, x):
        self.x_total += x
        self.S_total += np.outer(x, x)
        self.n_total += 1        

    def params(self):
        n = float(self.n)
        d = self.n_dims
        
        k0 = self.k0
        m0 = self.m0
        v0 = self.v0
        S0 = self.S0

        kn = k0 + n
        mn = (k0 * m0 + self.x) / kn
        vn = v0 + n
        Sn = S0 + self.S + k0 * np.outer(m0, m0) - kn * np.outer(mn, mn)
        
        Sigma = Sn / (vn + d + 2.)
        np.fill_diagonal(Sigma, np.maximum(Sigma.diagonal(), self.min_covar_diag))
        return mn, Sigma

    def expected_params(self):
        n = float(self.n_total)
        d = self.n_dims
        
        k0 = self.k0
        m0 = self.m0
        v0 = self.v0
        S0 = self.S0

        kn = float(k0 + n)
        mn = (k0 * m0 + self.x_total) / kn
        vn = float(v0 + n)
        Sn = S0 + self.S_total + k0 * np.outer(m0, m0) - kn * np.outer(mn, mn)
        
        Sigma = Sn / (vn + d + 2.)
        np.fill_diagonal(Sigma, np.maximum(Sigma.diagonal(), self.min_covar_diag))
        return mn, Sigma

    def generate_sample(self, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        mu, Sigma = frozen_params
        return np.atleast_1d(multivariate_normal.rvs(mean=mu, cov=Sigma))

    def marginal_likelihood(self, log=True, normalize=True):
        n = float(self.n)
        d = self.n_dims
        
        k0 = self.k0
        m0 = self.m0
        v0 = self.v0
        S0 = self.S0

        kn = float(k0 + n)
        mn = (k0 * m0 + self.x) / kn
        vn = float(v0 + n)
        Sn = S0 + self.S + k0 * np.outer(m0, m0) - kn * np.outer(mn, mn)
        
        lp = -0.5 * n * d * np.log(np.pi)
        for i in range(1, d + 1):
            lp += gammaln(0.5 * (vn + 1 - i)) - gammaln(0.5 * (v0 + 1 - i))
        # lp += gammaln(0.5 * vn)
        # lp -= gammaln(0.5 * v0)
        lp += 0.5 * d * np.log(k0)
        lp -= 0.5 * d * np.log(kn)
        # lp += 0.5 * v0 * np.log(np.linalg.det(S0))
        # lp -= 0.5 * vn * np.log(np.linalg.det(Sn))
        lp += 0.5 * v0 * np.linalg.slogdet(S0)[1]
        lp -= 0.5 * vn * np.linalg.slogdet(Sn)[1]
        
        if not np.isfinite(lp):
            print lp
            print 0.5 * n * d * np.log(np.pi)
            print gammaln(0.5 * vn)
            print gammaln(0.5 * v0)
            print 0.5 * d * np.log(k0)
            print 0.5 * d * np.log(kn)
            print 0.5 * v0 * np.linalg.slogdet(S0)[1]
            print 0.5 * vn * np.linalg.slogdet(Sn)[1]
            raise ValueError('log probability not finite')

        return lp if log else np.exp(lp)

    def vector(self, obs, log=True, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        mu, Sigma = frozen_params
        lpvec = multivariate_normal.logpdf(obs, mean=mu, cov=Sigma)
        # TODO: Getting underflow isseus above and traced it to here.
        # Haven't figured out a way to avoid it without this hack.
        # Note1: Added the min_covar_diag parameter, mimicking sklearn.
        # So far this seems to fix the issue.
        # Note2: ran into another stability issue. If lp is really small
        # than exp(lp) = 0, which messes up a bunch of upstream computation.
        # Quick fix is:
        lpvec = np.maximum(lpvec, -700)
        return lpvec if log else np.exp(lpvec)


class NIW(object):
    def __init__(self, n_states, n_dims, k0, m0, v0, S0, min_covar_diag=0.01):
        self.n_dims = n_dims
        m0 = np.array(m0)
        S0 = np.array(S0)
        self.n_states = n_states
        if m0.ndim == 1:
            m0 = [m0.copy() for i in range(n_states)]
        if S0.ndim == 2:
            S0 = [S0.copy() for i in range(n_states)]

        self.components = [NIWUnit(n_dims, k0, m0[i], v0, S0[i], min_covar_diag) for i in range(n_states)]

        self.k0 = k0
        self.m0 = m0
        self.v0 = v0
        self.S0 = S0
        self.min_covar_diag = min_covar_diag

    def reset(self):
        for component in self.components:
            component.reset()

    def posterior(self, weight=1.0):
        new = NIW(self.n_states, self.k0, self.m0, self.v0, self.S0, self.min_covar_diag)
        new.components = [component.posterior(weight) for component in self.components]
        return new

    def slice(self, i, j):
        new = NIW(j - i, self.k0, self.m0, self.v0, self.S0, self.min_covar_diag)
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
