import numpy as np
import itertools
from nose.tools import assert_true, assert_equal, assert_almost_equal
from .. import HMM, MarkovChain, DirCat, MVDirCat

def test_dircat_map():
    n_states = 5
    n_symbols = 10
    t_steps = 7
    
    initial_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states))
    transition_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states), n_states)
    emission_alpha = np.random.dirichlet(np.repeat(1. / n_symbols, n_symbols), n_states)
    
    markovchain = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)
    emission = DirCat(n_states, n_symbols, alpha=emission_alpha)

    hmm = HMM(markovchain, emission)
    obs = np.random.randint(0, n_symbols, t_steps)
    
    lp_max, states_max = hmm.map(obs, log=True)
    
    states_brute = None
    lp_brute = -np.inf
    for states in itertools.product(range(n_states), repeat=t_steps):
        lp = hmm.joint(states, obs, log=True)
        if lp > lp_brute:
            states_brute = states
            lp_brute = lp

    assert_almost_equal(lp_brute, lp_max)

    for s1, s2 in zip(states_max, states_brute):
        assert_equal(s1, s2)

def test_mvdircat_argmax_states():    
    n_states = 5
    n_dims = 3
    n_symbols = 10
    t_steps = 7
    
    initial_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states))
    transition_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states), n_states)
    emission_alpha = np.random.dirichlet(np.repeat(1. / n_symbols, n_symbols), (n_states, n_dims))
    
    markovchain = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)
    emission = MVDirCat(n_states, n_dims, n_symbols, alpha=emission_alpha)

    hmm = HMM(markovchain, emission)
    obs = np.random.randint(0, n_symbols, (t_steps, n_dims))
    
    lp_max, states_max = hmm.map(obs, log=True)
    
    states_brute = None
    lp_brute = -np.inf
    for states in itertools.product(range(n_states), repeat=t_steps):
        lp = hmm.joint(states, obs, log=True)
        if lp > lp_brute:
            states_brute = states
            lp_brute = lp

    assert_almost_equal(lp_brute, lp_max)

    for s1, s2 in zip(states_max, states_brute):
        assert_equal(s1, s2)

def test_dircat_likelihood():
    n_states = 5
    n_symbols = 10
    t_steps = 7
    
    initial_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states))
    transition_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states), n_states)
    emission_alpha = np.random.dirichlet(np.repeat(1. / n_symbols, n_symbols), n_states)
    
    markovchain = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)
    emission = DirCat(n_states, n_symbols, alpha=emission_alpha)

    hmm = HMM(markovchain, emission)
    obs = np.random.randint(0, n_symbols, t_steps)
    
    loglik = hmm.likelihood(obs, log=True)
    
    p = 0.0
    for states in itertools.product(range(n_states), repeat=t_steps):
        p += hmm.joint(states, obs, log=False)
    
    assert_almost_equal(loglik, np.log(p))

def test_mvdircat_likelihood():
    n_states = 5
    n_dims = 3
    n_symbols = 10
    t_steps = 7
    
    initial_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states))
    transition_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states), n_states)
    emission_alpha = np.random.dirichlet(np.repeat(1. / n_symbols, n_symbols), (n_states, n_dims))
    
    markovchain = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)
    emission = MVDirCat(n_states, n_dims, n_symbols, alpha=emission_alpha)

    hmm = HMM(markovchain, emission)
    obs = np.random.randint(0, n_symbols, (t_steps, n_dims))
    
    loglik = hmm.likelihood(obs, log=True)
    
    p = 0.0
    for states in itertools.product(range(n_states), repeat=t_steps):
        p += hmm.joint(states, obs, log=False)
    
    assert_almost_equal(loglik, np.log(p))

def test_propose():
    n_states = 5
    n_symbols = 10
    t_steps = 7
    
    initial_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states))
    transition_alpha = np.random.dirichlet(np.repeat(1. / n_states, n_states), n_states)
    emission_alpha = np.random.dirichlet(np.repeat(1. / n_symbols, n_symbols), n_states)
    
    markovchain = MarkovChain(n_states, initial_alpha=initial_alpha, transition_alpha=transition_alpha)
    emission = DirCat(n_states, n_symbols, alpha=emission_alpha)

    hmm = HMM(markovchain, emission)
    obs = np.random.randint(0, n_symbols, t_steps)

    lp_s1, states = hmm.propose(obs, log=True)
    lp_s2 = hmm.proposal_prob(states, obs, log=True)
    assert_almost_equal(lp_s1, lp_s2)
