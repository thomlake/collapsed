import numpy as np
from scipy.special import gammaln
from nose.tools import assert_true
from .. utils import viewrepeat, logmvbeta

def test_utils_viewrepeat():
    x = np.array([1, 2, 3])
    assert_true((x == viewrepeat(x, 1)).all())

    y = np.array([1, 2, 3, 1, 2, 3]).reshape(2, 3)
    assert_true((y == viewrepeat(x, 2)).all())
    
    y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]).reshape(3, 3)
    assert_true((y == viewrepeat(x, 3)).all())

def test_logmvbeta_beta():
    n1, n2, n3 = 5, 7, 8
    c = 10

    F = c * np.random.random((n1, n2))
    b1 = logmvbeta(F)
    b2 = 0.0
    for f in F:
        b2 += gammaln(f).sum() - gammaln(f.sum())
    assert_true(np.allclose(b1, b2))

    F = c * np.random.random((n1, n2, n3))
    b1 = logmvbeta(F)
    b2 = 0.0
    for Fi in F:
        for f in Fi:
            b2 += gammaln(f).sum() - gammaln(f.sum())
    assert_true(np.allclose(b1, b2))
