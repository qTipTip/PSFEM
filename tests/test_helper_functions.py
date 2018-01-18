import pytest
import numpy as np

from PSFEM.helper_functions import local_to_global


@pytest.mark.degrees_of_freedom
def test_local_to_global_numbering():
    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    connectivity_matrix = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])

    computed_dofs, computed_edge_dof_map, computed_vertex_dof_map = local_to_global(vertices, connectivity_matrix)
    expected_dofs = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        1: [4, 5, 6, 12, 13, 14, 15, 16, 8, 9, 10, 7]
    }

    assert computed_dofs == expected_dofs
