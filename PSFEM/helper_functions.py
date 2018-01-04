def local_to_global(vertices, connectivity_matrix):
    """
    Computes the local to global map for the C^1 hermite global basis on PS12-split of a triangulation.
    :param np.ndarray vertices: array of points
    :param np.ndarray connectivity_matrix: array of vertex indices
    :return dict:
    """

    local_to_global = {}
    visited_vertices = {}
    visited_edges = {}
    global_dof_number = 0

    for k, t in enumerate(connectivity_matrix):

        edges = [tuple([int(t[i]), int(t[(i + 1) % 3])]) for i in range(3)]
        local_indices = []

        for edge in edges:
            if edge[0] not in visited_vertices:
                local_indices += [global_dof_number, global_dof_number + 1, global_dof_number + 2]
                visited_vertices[edge[0]] = [global_dof_number, global_dof_number + 1, global_dof_number + 2]
                global_dof_number += 3
            else:
                local_indices += visited_vertices[edge[0]]

            oriented_edge = tuple(sorted(edge))
            if oriented_edge not in visited_edges:
                local_indices.append(global_dof_number)
                visited_edges[oriented_edge] = global_dof_number
                global_dof_number += 1
            else:
                local_indices.append(visited_edges[oriented_edge])
        local_to_global[k] = local_indices

    return local_to_global
