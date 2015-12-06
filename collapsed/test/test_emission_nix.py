import numpy as np
from scipy.special import gammaln
from nose.tools import assert_true, assert_equal, assert_almost_equal
from .. import NIXUnit

np.random.seed(sum(map(ord, 'collapsed')))

def test_nix_posterior():
    n_samples, n_dims = 10, 3
    kappa_0 = 0.01
    mu_0 = np.zeros(n_dims)
    nu_0 = 3.0
    sigma2_0 = np.ones(n_dims)
    X = np.random.normal(0, 1, (n_samples, n_dims))
    
    x_bar = X.mean(0)
    x_sum = X.sum(0)
    s_sum = (X * X).sum(0)
    ss_diff = ((X - x_bar)**2).sum(0)

    nix = NIXUnit(n_dims, kappa_0, mu_0, nu_0, sigma2_0, min_covar_diag=0)

    for x in X:
        nix.observe(x)

    post = nix.posterior()

    kappa_n = kappa_0 + n_samples
    mu_n = (kappa_0 * mu_0 + n_samples * x_bar) / (kappa_0 + n_samples)
    nu_n = nu_0 + n_samples

    mu_coeff = ((n_samples * kappa_0) / (kappa_0 + n_samples))
    mu_diff = mu_coeff * (mu_0 - x_bar) * (mu_0 - x_bar)
    sigma2_n = (nu_0 * sigma2_0 + ss_diff + mu_diff) / nu_n

    assert_true(np.allclose(nix.x_sum, x_sum))
    assert_true(np.allclose(nix.s_sum, s_sum))
    assert_true(np.allclose(nix.s_sum - n_samples * x_bar * x_bar, ss_diff))

    assert_true(np.allclose(kappa_n, post.kappa_0))
    assert_true(np.allclose(mu_n, post.mu_0))
    assert_true(np.allclose(nu_n, post.nu_0))
    assert_true(np.allclose(sigma2_n, post.sigma2_0))

    loglike = 0.0
    for i in range(n_dims):
        lli = 0.0
        lli += gammaln(0.5 * nu_n) - gammaln(0.5 * nu_0)
        lli += 0.5 * (np.log(kappa_0) - np.log(kappa_n))
        lli += 0.5 * nu_0 * (np.log(nu_0) + np.log(sigma2_0[i]))
        lli -= 0.5 * nu_n * (np.log(nu_n) + np.log(sigma2_n[i]))
        lli -= 0.5 * n_samples * np.log(np.pi)
        loglike += lli

    assert_true(np.allclose(loglike, nix.marginal_likelihood(log=True)))

