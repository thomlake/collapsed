import numpy as np
import simplex 
from utils import logmvbeta, flextile

class Mixture(object):
    def __init__(self, components, alpha=None, data=None):
        self.n_components = len(components)
        self.components = components

        if alpha is None:
            alpha = 1. / self.n_components

        self.alpha = flextile(alpha, (self.n_components,))
        self.counts = self.alpha.copy()
        self.totals = self.alpha.copy()
        
        self.n_accepted = 0
        self.n_attempts = 0
        self.burning = True
        
        # data
        self.data = []
        self.n = 0
        if data is not None:
            for k, states, obs in data:
                self.add_data(k, states, obs)

    def add_data(self, k, states, obs):
        self.observe(k, states, obs)
        self.data.append((k, states, obs))
        self.n += 1

    def marginal_likelihood(self, log=True, normalize=True):
        lf1 = logmvbeta(self.counts)
        if normalize:
            lf1 -= logmvbeta(self.alpha)
        lf2 = sum(component.marginal_likelihood(log=True, normalize=normalize) for component in self.components)
        return lf1 + lf2 if log else np.exp(lf1 + lf2)

    def observe(self, k, states, obs):
        self.components[k].observe(states, obs)
        self.counts[k] += 1.

    def forget(self, k, states, obs):
        self.components[k].forget(states, obs)
        self.counts[k] -= 1.

    def accumulate(self, k, states, obs):
        self.components[k].accumulate(states, obs)
        self.totals[k] += 1.

    def params(self):
        pi = simplex.project(self.counts)
        return {
            'mix': {
                'log': np.log(pi),
                'original': pi,
            },
            'components': [component.params() for component in self.components]
        }

    def expected_params(self):
        pi = simplex.project(self.totals)
        return {
            'mix': {
                'log': np.log(pi),
                'original': pi,
            },
            'components': [component.expected_params() for component in self.components]
        }

    def map(self, obs, log=True):
        scores = np.log(simplex.project(np.array(self.counts)))
        states = []
        for k, component in enumerate(self.components):
            score_k, states_k = component.map(obs, log=True)
            scores[k] += score_k
            states.append(states_k)
        k_max = scores.argmax()
        return scores[k_max] if log else np.exp(scores[k_max]), k_max, states[k_max]

    def propose(self, obs, log=True):
        probs = simplex.project(self.counts)
        k = probs.cumsum().searchsorted(np.random.random())
        lp_states, states = self.components[k].propose(obs)
        lp = np.log(probs[k]) + lp_states
        return lp if log else np.exp(lp), k, states

    def proposal_prob(self, k, states, obs, log=True):
        lp = np.log(simplex.project(self.counts)[k]) + self.components[k].proposal_prob(states, obs, log=True)
        return lp if log else np.exp(lp)

    def gibbs(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n

        for sample_number in xrange(n_samples):
            i = np.random.randint(self.n)
            k1, s1, obs = self.data[i]

            lp_k1_s1 = self.proposal_prob(k1, s1, obs, log=True)
            lp_k2_s2, k2, s2 = self.propose(obs, log=True)

            lp_D1 = self.marginal_likelihood(log=True, normalize=False)
            self.forget(k1, s1, obs)
            self.observe(k2, s2, obs)
            lp_D2 = self.marginal_likelihood(log=True, normalize=False)

            log_r = np.log(np.random.random())
            log_a = lp_D2 + lp_k1_s1 - lp_D1 - lp_k2_s2
            
            if not np.isfinite(log_a):
                msg = ['Error on sampling iteration {0}...'.format(sample_number)]
                msg.append('  log(a): {0}'.format(log_a))
                msg.append('  lp_D1: {0}'.format(lp_D1))
                msg.append('  lp_k1_s1: {0}'.format(lp_k1_s1))
                msg.append('  lp_D2: {0}'.format(lp_D2))
                msg.append('  lp_k2_s2: {0}'.format(lp_k2_s2))
                raise ValueError('\n'.join(msg))

            if log_r < log_a:
                self.data[i] = (k2, s2, obs)
                if not self.burning:
                    self.accumulate(k2, s2, obs)
                self.n_accepted += 1
            else:
                self.forget(k2, s2, obs)
                self.observe(k1, s1, obs)
                if not self.burning:
                    self.accumulate(k1, s1, obs)
            self.n_attempts += 1

    def acceptance_ratio(self):
        return self.n_accepted / float(self.n_attempts)
