import numpy as np
import pytest

from PSFEM.quadrature import midpoint_rule, gaussian_order_2_ps12


@pytest.mark.quadrature
def test_midpoint_quadratic_exact():

    def integrand(p):
        x, y = p
        return x**2 + x*y + y**2 + 1

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    expected_integral = 5/24 + 1/2
    computed_integral = midpoint_rule(integrand, vertices)

    np.testing.assert_approx_equal(computed_integral, expected_integral)

@pytest.mark.quadrature
def test_gaussian_quadratic():

    def integrand(p):
        x, y = p[:, 0], p[:, 1]
        return x**2 + x*y + y**2 + 1

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    expected_integral = 5/24 + 1/2
    computed_integral = gaussian_order_2_ps12(integrand, vertices)

    np.testing.assert_approx_equal(computed_integral, expected_integral)