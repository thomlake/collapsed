import numpy as np
from scipy.special import gammaln
from collections import defaultdict
import simplex
from utils import logmvbeta

class DPComponent(object):
    def __init__(self, model, k):
        self.model = model
        self.k = k
        self.count = 0.0

    def isempty(self):
        return self.count <= 0

    def observe(self, states, obs):
        self.model.observe(states, obs)
        self.count += 1.0

    def forget(self, states, obs):
        self.model.forget(states, obs)
        self.count -= 1.0
        if self.count < 0:
            raise ValueError('negative component count')

    def accumulate(self, states, obs):
        self.model.accumulate(states, obs)

    def params(self):
        return self.model.params()

    def expected_params(self):
        return self.model.expected_params()

    def marginal_likelihood(self, log=True, normalize=True):
        return self.model.marginal_likelihood(log=log, normalize=normalize)

    def likelihood(self, obs, log=True):
        return self.model.likelihood(obs, log=log)

    def map(self, obs, log=True):
        return self.model.map(obs, log=log)

    def proposal_prob(self, states, obs, log=True):
        return self.model.proposal_prob(states, obs, log=log)

    def propose(self, obs, log=True):
        return self.model.propose(obs, log=log)

class DP(object):
    def __init__(self, base, alpha=1.0):
        self.base = base
        self.alpha = float(alpha)

        self.components = []
        self.data = []

        self.n_accepted = 0
        self.n_attempts = 0
        self.burning = True

        self.n = 0

    def collect_garbage(self):
        empty_indices = [component.k for component in self.components if component.count == 0]
        component_total_count = sum(component.count for component in self.components)

        indexes = {component.k for component in self.components}
        if len(indexes) != len(self.components):
            raise ValueError('DP desynchronized: duplicate indexes')

        if component_total_count != self.n:
            raise ValueError('DP desynchronized: total {0} != {1} n'.format(component_total_count, self.n))

        if len(empty_indices) > 1:
            raise ValueError('DP desynchronized: more than 1 empty component')

        if len(empty_indices) > 0:
            k_pop = empty_indices.pop()
            self.components.pop(k_pop)
            for component in self.components:
                if component.k > k_pop:
                    component.k -= 1
            self.data = [(k - 1 if k > k_pop  else k, states, obs) for k, states, obs in self.data]

    def add_data(self, k, states, obs):
        self.observe(k, states, obs)
        self.data.append((k, states, obs))
        self.n += 1

    def observe(self, k, states, obs):
        self.components[k].observe(states, obs)

    def forget(self, k, states, obs):
        self.components[k].forget(states, obs)

    def accumulate(self, k, states, obs):
        self.components[k].accumulate(states, obs)

    def params(self):
        return {k: component.params() for k, component in self.components.items()}

    def expected_params(self):
        return {k: component.expected_params() for k, component in self.components.items()}

    def likelihood(self, obs, log=True):
        mix = simplex.project(self.component_counts())
        scores = [np.log(p) + component.likelihood(obs, log=True) for p, component in zip(mix, self.components)]
        ll = max(scores)
        return ll if log else np.exp(ll)

    def loglikelihood_vector(self, obs):
        logmix = np.log(simplex.project(self.component_counts()))
        scores = [lp + component.likelihood(obs, log=True) for lp, component in zip(logmix, self.components)]
        return np.array(scores)

    def map(self, obs, log=True):
        scores = np.log(simplex.project(self.component_counts()))
        states = []
        for k, component in enumerate(self.components):
            score_k, states_k = component.map(obs, log=True)
            scores[k] += score_k
            states.append(states_k)
        k_max = scores.argmax()
        return scores[k_max] if log else np.exp(scores[k_max]), k_max, states[k_max]

    def component_counts(self):
        return np.array([component.count for component in self.components])

    def marginal_likelihood(self, log=True, normalize=True):
        # CRP Term cancels. Suggested by Sinead
        lp = 0.0
        for component in self.components:
            if component.isempty():
                continue
            lp += component.marginal_likelihood(log=True, normalize=normalize)
        return lp if log else np.exp(lp)

    def propose(self, obs, log=True, adjust=False):
        n = float(self.n + self.alpha - adjust)
        r = n * np.random.random()

        for component in self.components:
            if r < component.count:
                lp_component = np.log(component.count) - np.log(n)
                lp_states, states = component.propose(obs, log=True)
                lp = lp_component + lp_states
                return lp if log else np.exp(lp), component.k, states
            r -= component.count
        component = DPComponent(self.base(), len(self.components))
        self.components.append(component)
        lp_component = np.log(self.alpha) - np.log(n)
        lp_states, states = component.propose(obs, log=True)
        lp = lp_component + lp_states
        return lp if log else np.exp(lp), component.k, states

    def proposal_prob(self, k, states, obs, log=True):
        n = self.n + self.alpha - 1.0
        component = self.components[k]
        if component.isempty():
            lp_component = np.log(self.alpha) - np.log(n)
        else:
            lp_component = np.log(component.count) - np.log(n)

        lp = lp_component + component.proposal_prob(states, obs, log=True)
        return lp if log else np.exp(lp)

    def gibbs(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        for i in indices:
            lp_D1 = self.marginal_likelihood(log=True, normalize=False)
            k1, s1, obs = self.data[i]
            self.forget(k1, s1, obs)

            lp_k1_s1 = self.proposal_prob(k1, s1, obs, log=True)
            lp_k2_s2, k2, s2 = self.propose(obs, adjust=True)
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
            self.collect_garbage()

    def acceptance_ratio(self):
        return self.n_accepted / float(self.n_attempts)
