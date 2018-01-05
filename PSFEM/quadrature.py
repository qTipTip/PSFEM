from SSplines import area, sub_triangles


def midpoint_rule(integrand, vertices):
    """
    Computes a numerical approximation to the integral over the triangle delineated by supplied vertices, using
    a two dimensional mid-point rule. Should integrate quadratic functions exactly.
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
    for sub_triangle in sub_triangles(vertices):
        integral += midpoint_rule(integrand, sub_triangle)
    return integral
