import copy
import numpy as np
from scipy.special import gammaln
from .. import simplex 
from .. utils import logmvbeta, viewrepeat, flextile

class MVDirCat(object):
    def __init__(self, n_states, n_dims, n_symbols, alpha=None):        
        self.n_states = n_states
        self.n_dims = n_dims
        self.n_symbols = n_symbols
        
        self.states = np.arange(n_states)
        self.dims = np.arange(n_dims)

        if alpha is None:
            alpha = np.tile(1. / n_symbols, (n_states, n_dims, n_symbols))
        elif isinstance(alpha, float):
            alpha = np.tile(alpha, (n_states, n_dims, n_symbols))

        alpha = np.array(alpha).astype(float)

        if alpha.shape != (n_states, n_dims, n_symbols):
            raise ValueError('alpha has wrong shape: {0}'.format(alpha.shape))

        self.alpha = alpha
        self.reset()

    def reset(self):
        self.counts = self.alpha.copy()
        self.totals = self.alpha.copy()

    def posterior(self, weight=1.0):
        return MVDirCat(self.n_states, self.n_dims, self.n_symbols, weight * self.counts)

    def slice(self, i, j):
        n_states = j - i
        new = MVDirCat(n_states, self.n_dims, self.n_symbols, alpha=self.alpha[i:j].copy())
        new.counts = self.counts[i:j].copy()
        new.totals = self.totals[i:j].copy()
        return new

    def observe(self, states, obs):
        T = len(states)
        states_r = viewrepeat(states, self.n_dims, axis=1).reshape(-1)
        dims_r = viewrepeat(self.dims, T, axis=0).reshape(-1)
        np.add.at(self.counts, (states_r, dims_r, obs.reshape(-1)), 1)

    def forget(self, states, obs):
        T = len(states)
        states_r = viewrepeat(states, self.n_dims, axis=1).reshape(-1)
        dims_r = viewrepeat(self.dims, T, axis=0).reshape(-1)
        np.add.at(self.counts, (states_r, dims_r, obs.reshape(-1)), -1)

    def accumulate(self, states, obs):
        T = len(states)
        states_r = viewrepeat(states, self.n_dims, axis=1).reshape(-1)
        dims_r = viewrepeat(self.dims, T, axis=0).reshape(-1)
        np.add.at(self.totals, (states_r, dims_r, obs.reshape(-1)), 1)

    def params(self):
        params = simplex.project(self.counts)
        return {'log': np.log(params), 'original': params}

    def expected_params(self):
        params = simplex.project(self.totals)
        return {'log': np.log(params), 'original': params}

    def marginal_likelihood(self, log=True, normalize=True):
        lp = logmvbeta(self.counts)
        if normalize:
            lp -= logmvbeta(self.alpha)
        return lp if log else np.exp(lp)

    def generate_sample(self, state, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        params = frozen_params['original']
        params = params[state]
        x = []
        for p in params:
            x.append(p.cumsum().searchsorted(np.random.random()))
        return np.array(x)

    def matrix(self, obs, log=True, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        log_params = frozen_params['log']
        T = len(obs)
        dims_r = viewrepeat(self.dims, T, axis=0)
        logB = log_params[:, dims_r, obs].sum(-1).T
        return logB if log else np.exp(logB)

