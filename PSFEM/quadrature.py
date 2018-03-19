import numpy as np
import quadpy as quadpy
from SSplines import area, ps12_sub_triangles, gaussian_quadrature, gaussian_quadrature_data, \
    points_from_barycentric_coordinates


def midpoint_rule(integrand, vertices):
    """
    Computes a numerical approximation to the integral over the triangle delineated by supplied vertices, using
    a two dimensional mid-point rule. Integrates quadratic functions exactly.
    :param callable integrand: function to integrate
    :param np.ndarray vertices: vertices of triangle
    :return: numerical approximation to integral.
    """

    integral = 0
    a = area(vertices)
    for i in range(3):
        mid_point = (vertices[i] + vertices[(i+1) % 3]) / 2
        integral += integrand(mid_point)
    return integral * a / 3


def midpoint_rule_ps12(integrand, vertices):
    """
    Computes a numerical approximation to the integral over the triangle delineated by supplied vertices, using
    a two dimensional mid-point rule for each of the twelve sub-triangles in the PS12-split. Useful for piecewise
    functions.
    :param callable integrand: function to integrate
    :param np.ndarray vertices: vertices of triangle
    :return: numerical approximation to integral
    """

    integral = 0
    for sub_triangle in ps12_sub_triangles(vertices):
        integral += midpoint_rule(integrand, sub_triangle)
    return integral


def gaussian_order_3_ps12(integrand, vertices):
    i = 0
    b, w = gaussian_quadrature_data(3)
    for sub_triangle in ps12_sub_triangles(vertices):
        i += gaussian_quadrature(sub_triangle, integrand, b, w)
    return i


def gaussian_order_2_ps12(integrand, vertices):
    i = 0
    b, w = gaussian_quadrature_data(2)
    for sub_triangle in ps12_sub_triangles(vertices):
        i += gaussian_quadrature(sub_triangle, integrand, b, w)
    return i


def quadpy_full(integrand, vertices, integration_scheme=quadpy.triangle.SevenPoint()):
    return quadpy.triangle.integrate(integrand, vertices.T, integration_scheme)


def compute_local_matrix(local_basis, triangle, b, w):
    p = points_from_barycentric_coordinates(triangle, b)
    m = np.atleast_3d(np.array([b.lapl(p) for b in local_basis]))
    m = np.swapaxes(m, 0, 1)
    M = np.einsum('...ik,kj...->...ij', m, m.T)

    A = np.sum(w[:, None, None] * M, axis=0)

    return A


def compute_local_matrix_ps12(local_basis, triangle, b, w):
    """
    Using a gaussian quadrature rule of given order, compute the local 12x12 matrix for the finite element
    solver.
    :param local_basis: set of 12 basis functions
    :param order: integration order
    :return: 12 x 12 local matrix
    """
    A = np.zeros((12, 12))
    for t in ps12_sub_triangles(triangle):
        A += area(t) * compute_local_matrix(local_basis, t, b, w)

    return A


def compute_local_vector(local_basis, func, triangle, b, w):
    p = points_from_barycentric_coordinates(triangle, b)
    m = np.atleast_3d(np.array([b(p) for b in local_basis]))
    m = np.swapaxes(m, 0, 1)
    f = func(p)
    M = np.einsum('Bjk,B->Bjk', m, f)
    A = np.sum(w[:, None, None] * M, axis=0)
    return A


def compute_local_vector_ps12(local_basis, f, triangle, b, w):
    """
    Using
    :param local_basis: set of 12 basis functions
    :param f: right hand side of strong form
    :param triangle: vertices of triangle
    :param order: integration order
    :return: 12x1 local vector
    """
    A = np.zeros((12, 1))
    for t in ps12_sub_triangles(triangle):
        A += area(t) * compute_local_vector(local_basis, f, t, b, w)
    return A
