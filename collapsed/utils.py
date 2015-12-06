import base64
import numpy as np
from scipy.special import gammaln

def viewrepeat(x, n, axis=0):
    if x.ndim != 1:
        raise ValueError('Can only view repeat 1 dimensional arrays [got: {0}]'.format(x.ndim))
    if axis == 0:
        return np.lib.stride_tricks.as_strided(x, (n, x.size), (0, x.itemsize))
    else:
        return np.lib.stride_tricks.as_strided(x, (x.size, n), (x.itemsize, 0))

def numpy_array_encode(obj):
    data_b64 = base64.b64encode(obj.data)
    return dict(data=data_b64,
                dtype=str(obj.dtype),
                shape=obj.shape)

def numpy_array_decode(d):
    data = base64.b64decode(d['data'])
    return np.frombuffer(data, d['dtype']).reshape(d['shape'])

def flextile(x, shape):
    if np.isscalar(x):
        return np.tile(x, shape)
    ndim = len(shape)
    head = shape[:-x.ndim]
    tail = shape[-x.ndim:]
    if tail != x.shape:
        raise ValueError('Trailing dimension mismatch [got: {0}, requested: {1}]'.format(x.shape, shape))
    return np.tile(x, list(head) + [1] * len(tail))

def logmvbeta(c):
    return (gammaln(np.where(c > 0, c, 1)).sum(axis=-1) - gammaln(c.sum(axis=-1))).sum()