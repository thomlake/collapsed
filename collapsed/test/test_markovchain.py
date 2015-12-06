import numpy as np
from nose.tools import assert_true, assert_equal, assert_almost_equal
from .. import MarkovChain

np.random.seed(sum(map(ord, 'collapsed')))

def test_markovchain():
    n_states = 6
    n_samples = 10

    dist = MarkovChain(n_states)

    assert_true(np.allclose(dist.params()['log'][0], np.log(1. / n_states)))
    assert_true(np.allclose(dist.params()['log'][1], np.log(1. / n_states)))

    assert_true(np.allclose(dist.params()['original'][0], 1. / n_states))
    assert_true(np.allclose(dist.params()['original'][1], 1. / n_states))

    transition_counts = dist.transition_counts.copy()
    initial_counts = dist.initial_counts.copy()

    states = np.random.randint(0, n_states, n_samples)

    dist.observe(states)
    initial_counts[states[0]] += 1
    for stm1, st in zip(states[:-1], states[1:]):
        transition_counts[stm1, st] += 1
    assert_true(np.allclose(transition_counts, dist.transition_counts))
    assert_true(np.allclose(initial_counts, dist.initial_counts))

    dist.forget(states)
    initial_counts[states[0]] -= 1
    for stm1, st in zip(states[:-1], states[1:]):
        transition_counts[stm1, st] -= 1
    assert_true(np.allclose(transition_counts, dist.transition_counts))
    assert_true(np.allclose(initial_counts, dist.initial_counts))

    assert_true(np.allclose(dist.params()['log'][0], np.log(1. / n_states)))
    assert_true(np.allclose(dist.params()['log'][1], np.log(1. / n_states)))

    assert_true(np.allclose(dist.params()['original'][0], 1. / n_states))
    assert_true(np.allclose(dist.params()['original'][1], 1. / n_states))
