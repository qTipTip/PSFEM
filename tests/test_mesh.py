import numpy as np
import pytest

from PSFEM.mesh import Mesh


def test_incident_triangles():

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5]
    ])

    triangles = np.array([
        [0, 1, 4],
        [1, 3, 4],
        [3, 2, 4],
        [2, 0, 4]
    ])

    M = Mesh(vertices, triangles)

    expected_triangles = [0, 1, 2, 3]
    computed_triangles = M.incident_triangles(4)

    np.testing.assert_array_almost_equal(computed_triangles, expected_triangles)

    expected_triangles = [2, 3]
    computed_triangles = M.incident_triangles(2)

    np.testing.assert_array_almost_equal(computed_triangles, expected_triangles)


def test_adjacent_triangles():

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5]
    ])

    triangles = np.array([
        [0, 1, 4],
        [1, 3, 4],
        [3, 2, 4],
        [2, 0, 4]
    ])

    M = Mesh(vertices, triangles)

    expected_triangles = [2, 3]
    computed_triangles = M.adjacent_triangles(7)

    np.testing.assert_array_almost_equal(computed_triangles, expected_triangles)

    expected_triangles = [1, 2]
    computed_triangles = M.adjacent_triangles(6)

    np.testing.assert_array_almost_equal(computed_triangles, expected_triangles)

    expected_triangles = [0]
    computed_triangles = M.adjacent_triangles(0)

    np.testing.assert_array_almost_equal(computed_triangles, expected_triangles)

