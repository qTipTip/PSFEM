import numpy as np
import quadpy as quadpy
from SSplines import area, ps12_sub_triangles, gaussian_quadrature, gaussian_quadrature_data


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


