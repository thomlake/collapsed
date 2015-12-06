import numpy as np
from nose.tools import assert_true, assert_equal, assert_almost_equal
from .. import DirCat, MVDirCat, NIWUnit, NIW

np.random.seed(sum(map(ord, 'collapsed')))

def test_emission_dircat():
    n_states = 3
    n_symbols = 5
    n_samples = 10

    dist = DirCat(n_states, n_symbols)

    assert_true(np.allclose(dist.params()['log'], np.log(1. / n_symbols)))
    assert_true(np.allclose(dist.params()['original'], 1. / n_symbols))

    counts = dist.counts.copy()
    states = np.random.randint(0, n_states, n_samples)
    obs = np.random.randint(0, n_symbols, n_samples)

    dist.observe(states, obs)
    for s, o in zip(states, obs):
        counts[s,o] += 1
    assert_true(np.allclose(counts, dist.counts))

    dist.forget(states, obs)
    for s, o in zip(states, obs):
        counts[s, o] -= 1
    assert_true(np.allclose(counts, dist.counts))

    assert_true(np.allclose(dist.params()['log'], np.log(1. / n_symbols)))
    assert_true(np.allclose(dist.params()['original'], 1. / n_symbols))

def test_emission_mvdircat():
    n_states = 3
    n_dims = 6
    n_symbols = 5
    n_samples = 10

    dist = MVDirCat(n_states, n_dims, n_symbols)

    assert_true(np.allclose(dist.params()['log'], np.log(1. / n_symbols)))
    assert_true(np.allclose(dist.params()['original'], 1. / n_symbols))

    counts = dist.counts.copy()
    
    states = np.random.randint(0, n_states, n_samples)
    obs = np.random.randint(0, n_symbols, (n_samples, n_dims))

    dist.observe(states, obs)
    for s, o in zip(states, obs):
        for i in range(n_dims): 
            counts[s, i, o[i]] += 1
    assert_true(np.allclose(counts, dist.counts))

    dist.forget(states, obs)
    for s, o in zip(states, obs):
        for i in range(n_dims): 
            counts[s, i, o[i]] -= 1
    assert_true(np.allclose(counts, dist.counts))

    assert_true(np.allclose(dist.params()['log'], np.log(1. / n_symbols)))
    assert_true(np.allclose(dist.params()['original'], 1. / n_symbols))

def test_niw_unit():
    d = 3
    n = 10
    k0 = 0.
    m0 = np.zeros(d)
    S0 = np.zeros((d, d))
    v0 = d

    X = np.random.normal(0, 1, (n, d))
    m1 = X.mean(0)
    S1 = np.cov(X, rowvar=0, bias=1)

    niw = NIWUnit(k0, m0, v0, S0)
    for x in X:
        niw.observe(x)

    m2, S2 = niw.params()
    assert_true(np.allclose(m1, m2))
    assert_true(np.allclose(S1, S2))
