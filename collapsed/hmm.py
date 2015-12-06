import copy
import numpy as np
import simplex

class HMM(object):
    def __init__(self, markovchain, emission, data=None, metadata=None):
        self.n_states = markovchain.n_states

        # factors
        self.markovchain = markovchain
        self.emission = emission

        # sampler stuff
        self.n_accepted = 0
        self.n_attempts = 0
        self.burning = True

        # data
        self.data = []
        self.n = 0
        if data is not None:
            for states, obs in data:
                self.add_data(states, obs)
        self.metadata = {} if metadata is None else metadata

    def reset(self):
        self.markovchain.reset()
        self.emission.reset()

    def posterior(self, weight=1.0):
        mc_post = self.markovchain.posterior(weight)
        em_post = self.emission.posterior(weight)
        return HMM(mc_post, em_post, self.metadata)

    def slice(self, i, j):
        return HMM(self.markovchain.slice(i, j),
                   self.emission.slice(i, j),
                   metadata=copy.deepcopy(self.metadata))

    def add_data(self, states, obs, observe=True):
        if observe:
            self.observe(states, obs)
        self.data.append((states, obs))
        self.n += 1

    def observe(self, states, obs):
        self.markovchain.observe(states)
        self.emission.observe(states, obs)

    def forget(self, states, obs):
        self.markovchain.forget(states)
        self.emission.forget(states, obs)

    def accumulate(self, states, obs):
        self.markovchain.accumulate(states)
        self.emission.accumulate(states, obs)

    def params(self):
        frozen_params = {}
        frozen_params['markovchain'] = self.markovchain.params()
        frozen_params['emission'] = self.emission.params()
        return frozen_params

    def expected_params(self):
        frozen_params = {}
        frozen_params['markovchain'] = self.markovchain.expected_params()
        frozen_params['emission'] = self.emission.expected_params()
        return frozen_params

    def marginal_likelihood(self, log=True, normalize=True):
        lf1 = self.markovchain.marginal_likelihood(log=True, normalize=normalize)
        lf2 = self.emission.marginal_likelihood(log=True, normalize=normalize)
        return lf1 + lf2 if log else np.exp(lf1 + lf2)

    ######################
    # Inference Routines #
    ######################
    def joint(self, states, obs, log=True, frozen_params=None):
        """Compute log joint probability of an (obs, states) 
        pair from the current fixed_params

        Parameters
        ----------
        states: np.ndarray
            States
        obs: np.ndarray
            Observations
        log: bool
            If True return log joint probability
        frozen_params: dict (optional)
            Frozen parameters.

        Returns
        -------
        score: float
            P(X=obs, S=states)
        """
        if frozen_params is None:
            frozen_params = self.params()
        log_pi, log_A = frozen_params['markovchain']['log']
        log_B = self.emission.matrix(obs, log=True, frozen_params=frozen_params['emission'])

        T = log_B.shape[0]
        lp = log_pi[states[0]] + log_B[0, states[0]]
        for t in xrange(1, T):
            lp += log_A[states[t - 1], states[t]] + log_B[t, states[t]]
        return lp if log else np.exp(lp)

    def likelihood(self, obs, log=True, frozen_params=None):
        """Compute the likelihood of an observation sequence
        by marginalizing over states.

        Parameters
        ----------
        obs: np.ndarray
            Observations
        log: bool
            If True return log likelihood
        frozen_params: dict (obtional)
            Frozen parameters.
        
        Returns
        -------
        score: float
            logP(X=obs) if log esle P(X=obs)
        """
        if frozen_params is None:
            frozen_params = self.params()
        pi, A = frozen_params['markovchain']['original']
        B = self.emission.matrix(obs, log=False, frozen_params=frozen_params['emission'])

        ll = 0.
        T = B.shape[0]
        
        p = pi * B[0]
        c = 1. / p.sum()
        p = p * c
        ll += np.log(c)
        for t in xrange(1, T):
            p = np.dot(p, A) * B[t]
            c = 1. / p.sum()
            p = p * c
            ll += np.log(c)
        return -ll if log else np.exp(-ll)

    def map(self, obs, log=True, frozen_params=None):
        """Find the MAP state sequence given obs.

        Parameters
        ----------
        obs: np.ndarray
            Observations
        log: bool
            If True return log probability
        frozen_params: dict (optional)
            Frozen parameters.

        Returns
        -------
        score: np.ndarray
            score = P(X=obs,S=s_max)
        s_max: np.ndarray
            argmax_s { logp(X=obs, S=s) }
        """
        if frozen_params is None:
            frozen_params = self.params()
        log_pi, log_A = frozen_params['markovchain']['log']
        log_B = self.emission.matrix(obs, log=True, frozen_params=frozen_params['emission'])

        n_states = log_A.shape[0]
        state_index = np.arange(n_states)
        T = log_B.shape[0]
        back_ptrs = np.empty((T, n_states), dtype=int)
        lp_table = np.empty((T, n_states), dtype=float)
        
        lp_table[0] = log_pi + log_B[0]
        for t in xrange(1, T):
            logp_state_trans = log_A.T + lp_table[t-1]
            back_ptrs[t] = logp_state_trans.argmax(axis=1)
            lp_table[t] = logp_state_trans[state_index, back_ptrs[t]] + log_B[t]
        
        states_max = [lp_table[-1].argmax()]
        for t in range(T - 1, 0, -1):
            states_max.append(back_ptrs[t, states_max[-1]])

        # need to reverse the state array
        states_max = np.array(states_max, dtype=int)[::-1].copy()
        score = lp_table[-1, states_max[-1]]
        return score if log else np.exp(score), states_max

    def forward_matrices(self, obs, log=True, frozen_params=None):
        """Compute forward log probability matrices.

        Parameters
        ----------
        obs: np.ndarray
            Observations
        log: bool
            If True return log probability matrices
        frozen_params: dict (optional)
            Frozen parameters.

        Returns
        -------
        Pmats: list of np.ndarray
            Pmats[t][i,j] = P(S[t]=i, S[t-1]=j| X=obs[0:t])
        """
        if frozen_params is None:
            frozen_params = self.params()
        pi, A = frozen_params['markovchain']['original']
        B = self.emission.matrix(obs, log=False, frozen_params=frozen_params['emission'])

        T = B.shape[0]
        n = A.shape[0]

        P0 = pi * B[0]
        P0 = ptm1 = P0 / P0.sum()
        Pmats = [P0]

        for t in range(1, T):
            Pt = B[t].repeat(n).reshape(n, n)
            Pt *= A.T
            Pt *= ptm1
            Pt = Pt / Pt.sum()
            Pmats.append(Pt)
            ptm1 = Pt.sum(1)
        
        return map(np.log, Pmats) if log else Pmats

    def fbsample(self, obs, log=True, frozen_params=None):
        """Sample a sequence of states given parameters.

        Parameters
        ----------
        obs: np.ndarray
            Observations
        log: bool
            If True return log probability of the sample
        frozen_params: dict (optional)
            Frozen parameters.

        Returns
        -------
        score: float
            P(S=states | X=obs)
        states: np.ndarray
            sampled states

        Notes
        -----
        @article{scott2002bayesian,
            title={Bayesian methods for hidden Markov models},
            author={Scott, Steven L},
            journal={Journal of the American Statistical Association},
            volume={97},
            number={457},
            year={2002}
        }
        """
        Pmats = self.forward_matrices(obs, log=False, frozen_params=frozen_params)
        T = len(Pmats)
        lp = 0.0
        states = []
        
        # p = np.exp(Pmats[T-1]).sum(1)
        p = Pmats[T-1].sum(1)
        i = p.cumsum().searchsorted(np.random.random())
        lp += np.log(p[i])
        states.append(i)

        for t in range(T - 1, 0, -1):
            i = states[-1]
            p = simplex.project(Pmats[t][i])
            j = p.cumsum().searchsorted(np.random.random())
            lp += np.log(p[j])
            states.append(j)
        return lp if log else np.exp(lp), np.array(states[::-1])

    def generate_sample(self, T, frozen_params=None):
        if frozen_params is None:
            frozen_params = self.params()
        states = self.markovchain.generate_sample(T, frozen_params=frozen_params['markovchain'])
        obs = [self.emission.generate_sample(state, frozen_params=frozen_params['emission']) for state in states]
        return np.array(states), np.array(obs)

    ####################
    # Viterbi Training #
    ####################
    def fit(self, trainX, validX=None, max_epochs=50, tol=1e-6, verbose=True):
        train_lp_epoch = -np.inf
        valid_lp_epoch = -np.inf

        for epoch in range(max_epochs):
            train_lp_epoch_prev, train_lp_epoch = train_lp_epoch, 0.0
            valid_lp_epoch_prev, valid_lp_epoch = valid_lp_epoch, 0.0
            params = self.params()
            self.reset()
            
            for obs in trainX:
                lp, states = self.map(obs, log=True, frozen_params=params)
                train_lp_epoch += lp
                self.observe(states, obs)

            if validX is not None:
                for obs in validX:
                    valid_lp_epoch += self.map(obs, log=True, frozen_params=params)[0]
                valid_lp_epoch /= len(validX)
                if valid_lp_epoch - valid_lp_epoch_prev < tol:
                    break
 
            train_lp_epoch /= len(trainX)
            if validX is None and train_lp_epoch - train_lp_epoch_prev < tol:
                break

            if verbose:
                msg = '[epoch: {0}] '.format(epoch)
                if validX is not None:
                    msg += 'logp(valid) = {0:0.04f}, '.format(valid_lp_epoch)
                msg += 'logp(train) = {0:0.04f}'.format(train_lp_epoch)
                print msg
        
        if verbose:
            msg = '[terminated after {0} epochs] '.format(epoch)
            if validX is not None:
                msg += 'logp(valid) = {0:0.04f}, '.format(valid_lp_epoch)
            msg += 'logp(train) = {0:0.04f}'.format(train_lp_epoch)
            print msg
        
        return self.params()

    ##############################################
    # Collapsed Metropolis-Hastings within Gibbs #
    ##############################################
    def propose(self, obs, log=True):
        return self.fbsample(obs, log=log)

    def proposal_prob(self, states, obs, log=True):
        score = self.joint(states, obs, log=True) - self.likelihood(obs, log=True) 
        return score if log else np.exp(score)

    def gibbs(self, n_samples=None):
        if n_samples is None:
            n_samples = self.n
        
        for sample_number in xrange(n_samples):
            lp_D1 = self.marginal_likelihood(log=True, normalize=False)

            i = np.random.randint(self.n)
            s1, obs = self.data[i]
            
            lp_s1 = self.proposal_prob(s1, obs, log=True)
            lp_s2, s2 = self.propose(obs, log=True)
            self.forget(s1, obs)
            self.observe(s2, obs)
            lp_D2 = self.marginal_likelihood(log=True, normalize=False)

            log_r = np.log(np.random.random())
            log_a = lp_D2 + lp_s1 - lp_D1 - lp_s2

            if not np.isfinite(log_a):
                msg = ['Error on sampling iteration {0}...'.format(sample_number)]
                msg.append('  log(a): {0}'.format(log_a))
                msg.append('  lp_D1: {0}'.format(lp_D1))
                msg.append('  lp_s1: {0}'.format(lp_s1))
                msg.append('  lp_D2: {0}'.format(lp_D2))
                msg.append('  lp_s2: {0}'.format(lp_s2))
                msg.append('  joint(s1): {0}'.format(self.joint(s1, obs, log=True)))
                msg.append('  joint(s2): {0}'.format(self.joint(s2, obs, log=True)))
                msg.append('  loglik: {0}'.format(self.likelihood(obs, log=True)))
                raise ValueError('\n'.join(msg))
                
            if log_r < log_a:
                self.data[i] = (s2, obs)
                if not self.burning:
                    self.accumulate(s2, obs)
                self.n_accepted += 1
            else:
                self.forget(s2, obs)
                self.observe(s1, obs)
                if not self.burning:
                    self.accumulate(s1, obs)
            self.n_attempts += 1

    def acceptance_ratio(self):
        return self.n_accepted / float(self.n_attempts)
