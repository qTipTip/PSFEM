import numpy as np

from PSFEM.mesh import Mesh


def local_to_global(vertices, connectivity_matrix):
    """
    Computes the local to global map for the C^1 hermite global basis on PS12-split of a triangulation.
    :param np.ndarray vertices: array of points
    :param np.ndarray connectivity_matrix: array of vertex indices
    :return dict:
    """

    local_to_global_map = {}
    visited_vertices = {}
    visited_edges = {}

    # the four following lists keep track of which mesh entity correspond to which type of dof
    value_dofs = {}
    dx_dofs = {}
    dy_dofs = {}
    edge_dofs = {}
    # these two dictionaries map global dof to edge or vertex.
    edge_dof_map = {}
    vertex_dof_map = {}

    global_dof_number = 0
    for k, t in enumerate(connectivity_matrix):

        edges = [tuple([int(t[i]), int(t[(i + 1) % 3])]) for i in range(3)]
        local_indices = []

        for edge in edges:
            if edge[0] not in visited_vertices:
                local_indices += [global_dof_number, global_dof_number + 1, global_dof_number + 2]
                visited_vertices[edge[0]] = [global_dof_number, global_dof_number + 1, global_dof_number + 2]

                value_dofs[edge[0]] = global_dof_number
                dx_dofs[edge[0]] = global_dof_number + 1
                dy_dofs[edge[0]] = global_dof_number + 2

                for vertex_dof in visited_vertices[edge[0]]:
                    vertex_dof_map[vertex_dof] = edge[0]
                global_dof_number += 3
            else:
                local_indices += visited_vertices[edge[0]]

            oriented_edge = tuple(sorted(edge))
            if oriented_edge not in visited_edges:
                local_indices.append(global_dof_number)
                visited_edges[oriented_edge] = global_dof_number
                edge_dof_map[global_dof_number] = oriented_edge
                edge_dofs[oriented_edge] = global_dof_number
                global_dof_number += 1
            else:
                local_indices.append(visited_edges[oriented_edge])
        local_to_global_map[k] = local_indices
    return local_to_global_map, edge_dof_map, vertex_dof_map, value_dofs, dx_dofs, dy_dofs, edge_dofs


def unit_square_uniform(n):
    vertices = np.array([
        [x, y]
        for x in np.linspace(0, 1, n + 1)
        for y in np.linspace(0, 1, n + 1)
    ])

    triangles = []
    for i in range(n):
        for j in range(n):
            triangles.append([(n + 1) * j + i, (n + 1) * j + i + 1, (n + 1) * j + i + (n + 1)])
            triangles.append([(n + 1) * j + i + 1, (n + 1) * j + 1 + (n + 1) + i, (n + 1) * j + i + (n + 1)])

    return Mesh(vertices, np.array(triangles))