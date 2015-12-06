import numpy as np
from nose.tools import assert_true, assert_equal, assert_almost_equal
from .. import simplex

def test_project_vector():
    n = 10
    x = simplex.project_vector(np.random.random(n))
    assert_almost_equal(x.sum(), 1)

    x = np.random.normal(0, 1, n)
    x[0] = np.random.random()
    x = simplex.project_vector(x)
    assert_almost_equal(x.sum(), 1)

def test_project_columns():
    n_rows, n_cols = 5, 10
    A = simplex.project_columns(np.random.random((n_rows, n_cols)))
    assert_true(np.allclose(A.sum(0), 1))

    A = np.random.normal(0, 1, (n_rows, n_cols))
    A[0] = np.random.random(n_cols) # ensure at least every column has 1 positive value
    A = simplex.project_columns(A)
    assert_true(np.allclose(A.sum(0), 1))

def test_project_rows():
    n_rows, n_cols = 5, 10
    A = simplex.project_rows(np.random.random((n_rows, n_cols)))
    assert_true(np.allclose(A.sum(1), 1))

    A = simplex.project_rows(np.random.normal(0, 1, (n_rows, n_cols)))
    A[:,0] = np.random.random(n_rows)
    A = simplex.project_rows(A)
    assert_true(np.allclose(A.sum(1), 1))

def test_project():
    n1, n2, n3 = 5, 10, 6
    x = np.random.normal(0, 1, n1)
    x[0] = np.random.random()
    x = simplex.project(x)
    assert_almost_equal(x.sum(), 1)

    A = np.random.normal(0, 1, (n1, n2))
    A[:,0] = np.random.random(n1)
    A = simplex.project(A)
    assert_true(np.allclose(A.sum(-1), 1))

    A = np.random.normal(0, 1, (n1, n2, n3))
    A[:,:,0] = np.random.random((n1, n2))
    A = simplex.project(A)
    assert_true(np.allclose(A.sum(-1), 1))

def test_vector():
    n = 10
    x = simplex.vector(n)
    assert_almost_equal(x.sum(), 1)

def test_column_matrix():
    n_rows, n_cols = 5, 10
    A = simplex.column_matrix(n_rows, n_cols)
    assert_true(np.allclose(A.sum(0), 1))

def test_row_matrix():
    n_rows, n_cols = 5, 10
    A = simplex.row_matrix(n_rows, n_cols)
    assert_true(np.allclose(A.sum(1), 1))

def test_array():
    n1, n2, n3 = 5, 10, 6
    x = simplex.array((n1,))
    assert_almost_equal(x.sum(), 1)

    A = simplex.array((n1, n2))
    assert_true(np.allclose(A.sum(-1), 1))

    A = simplex.array((n1, n2, n3))
    assert_true(np.allclose(A.sum(-1), 1))
