import numpy as np

import PSFEM


def f(p):
    """
    The exact source term for the model problem for the biharmonic equation.
    """
    x, y = p[:, 0], p[:, 1]
    return 8 * np.pi ** 4 * (np.sin(2 * np.pi * x) - 8 * np.sin(4 * np.pi * x) + 25 * np.sin(2 * np.pi * (x - 2 * y)) -
                             32 * np.sin(4 * np.pi * (x - y)) - np.sin(2 * np.pi * y) + 8 * (np.sin(4 * np.pi * y) +
                                                                                             np.sin(2 * np.pi * (
                                                                                             -x + y))) + 25 * np.sin(
        4 * np.pi * x - 2 * np.pi * y))


def a(u, v):
    """
    Given two basis splines on the PS12, computes the L2 inner product of their laplacians.
    """

    def lhs(p):
        return u.lapl(p.T) * v.lapl(p.T)

    return lhs


def L(v):
    """
    Given a source term f, computes the L2 inner product against a basis spline
    on the PS12.
    """

    def rhs(p):
        return v(p.T) * f(p.T)

    return rhs


def test_boundary_conditions():
    M = PSFEM.unit_square_uniform(1)
    V = PSFEM.CompositeSplineSpace(M)

    c0 = PSFEM.solve(a, L, V, dirichlet=lambda x: 0)
    c1 = PSFEM.solve(a, L, V, dirichlet=lambda x: 1)

    u0 = V.function(c0)
    u1 = V.function(c1)

    p = np.array([
        [x, y]
        for x in np.linspace(0, 1, 10)
        for y in np.linspace(0, 1, 10)
    ])

    vals = np.zeros(len(p))
    for i in range(len(vals)):
        vals[i] = u1(p[i]) - u0(p[i])
    np.testing.assert_almost_equal(vals, np.ones(len(p)))
