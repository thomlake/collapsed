import numpy as np

def project_vector(v):
    v = np.maximum(v, 0)
    return v / v.sum()

def project_columns(m):
    m = np.maximum(m, 0)
    return m / m.sum(0)

def project_rows(m):
    m = np.maximum(m, 0)
    return m / m.sum(1)[...,np.newaxis]

def project(m):
    m = np.maximum(m, 0)
    return m / m.sum(-1)[...,np.newaxis]

def vector(m, alpha=1.):
    if np.isscalar(alpha):
        alpha = np.repeat(alpha, m)
    if len(alpha) != m:
        raise ValueError('len(alpha) must equal number of elements [expected: {0}, got: {1}]'.format(m, len(alpha)))
    return np.random.dirichlet(alpha)

def column_matrix(m, n, alpha=1.):
    if np.isscalar(alpha):
        alpha = np.repeat(alpha, m)
    if len(alpha) != m:
        raise ValueError('len(alpha) must equal number of columns [expected: {0}, got: {1}]'.format(m, len(alpha)))
    return np.random.dirichlet(alpha, n).T.copy(order='C')

def row_matrix(m, n, alpha=1.):
    if np.isscalar(alpha):
        alpha = np.repeat(alpha, n)
    if len(alpha) != n:
        raise ValueError('len(alpha) must equal number of rows [expected: {0}, got: {1}]'.format(n, len(alpha)))
    return np.random.dirichlet(alpha, m)

def array(shape, alpha=1.):
    if np.isscalar(alpha):
        alpha = np.tile(alpha, shape[-1])
    if alpha.shape[0] != shape[-1]:
        raise ValueError('Error simplex.array({0}, {1})'.format(shape, alpha))
    return np.random.dirichlet(alpha, shape[:-1])
