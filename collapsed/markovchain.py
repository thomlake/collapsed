import numpy as np
import simplex 
from utils import logmvbeta, flextile


class MarkovChain(object):
    def __init__(self, n_states, initial_alpha=None, transition_alpha=None):
        self.n_states = n_states

        if initial_alpha is None:
            initial_alpha = np.tile(1. / n_states, (n_states,))
        elif isinstance(initial_alpha, float):
            initial_alpha = np.tile(initial_alpha, (n_states,))
        
        if transition_alpha is None:
            transition_alpha = np.tile(1. / n_states, (n_states, n_states))
        elif isinstance(transition_alpha, float):
            transition_alpha = np.tile(transition_alpha, (n_states, n_states))

        initial_alpha = np.array(initial_alpha).astype(float)
        transition_alpha = np.array(transition_alpha).astype(float)

        if initial_alpha.shape != (n_states,):
            raise ValueError('initial_alpha has wrong shape: {0}'.format(initial_alpha.shape))
        if transition_alpha.shape != (n_states, n_states):
            raise ValueError('transition_alpha has wrong shape: {0}'.format(transition_alpha.shape))

        self.initial_alpha = initial_alpha
        self.transition_alpha = transition_alpha

        self.reset()

    def reset(self):
        self.initial_counts = self.initial_alpha.copy()
        self.initial_totals = self.initial_alpha.copy()
        
        self.transition_counts = self.transition_alpha.copy()
        self.transition_totals = self.transition_alpha.copy()

    def posterior(self, weight=1.0):
        return MarkovChain(self.n_states, weight * self.initial_counts, weight * self.transition_counts)

    def slice(self, i, j):
        n_states = j - i
        initial_alpha = self.initial_alpha[i:j].copy()
        transition_alpha = self.transition_alpha[i:j,i:j].copy()
        new = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)

        new.initial_counts = self.initial_counts[i:j].copy()
        new.initial_totals = self.initial_totals[i:j].copy()
        new.transition_counts = self.transition_counts[i:j,i:j].copy()
        new.transition_totals = self.transition_totals[i:j,i:j].copy()

        return new

    def observe(self, states):
        self.initial_counts[states[0]] += 1
        np.add.at(self.transition_counts, (states[:-1], states[1:]), 1)

    def forget(self, states):
        self.initial_counts[states[0]] -= 1
        np.add.at(self.transition_counts, (states[:-1], states[1:]), -1)

    def accumulate(self, states):
        self.initial_totals[states[0]] += 1
        np.add.at(self.transition_totals, (states[:-1], states[1:]), 1)

    def params(self):
        pi = simplex.project(self.initial_counts)
        A = simplex.project(self.transition_counts)
        return {'log': (np.log(pi), np.log(A)), 'original': (pi, A)}

    def expected_params(self):
        pi = simplex.project(self.initial_totals)
        A = simplex.project(self.transition_totals)
        return {'log': (np.log(pi), np.log(A)), 'original': (pi, A)}

    def marginal_likelihood(self, log=True, normalize=True):
        lp = logmvbeta(self.initial_counts) + logmvbeta(self.transition_counts)
        if normalize:
            lp -= (logmvbeta(self.initial_alpha) + logmvbeta(self.transition_alpha))
        return lp if log else np.exp(lp)

    def generate_sample(self, T, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        pi, A = frozen_params['original']
        
        states = [pi.cumsum().searchsorted(np.random.random())]
        while len(states) < T:
            states.append(A[states[-1]].cumsum().searchsorted(np.random.random()))
        return states
