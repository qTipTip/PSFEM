import itertools
import numpy as np


class Mesh(object):
    """
    Given a set of vertices and a connectivity matrix describing each triangle,
    represents the corresponding mesh.
    """

    def __init__(self, vertices, connectivity_matrix):
        """
        Initialize a mesh with connectivity matrix and vertices.
        :param np.ndarray vertices: vertex coordinates
        :param np.ndarray connectivity_matrix: vertex indices
        """

        self.vertices = vertices
        self.triangles = connectivity_matrix

        self._generate_data()

    def vertex_to_triangles(self, vertex_id):
        pass

    def edge_to_triangles(self, edge_id):
        pass

    def interior_vertices(self):
        pass

    def interior_edges(self):
        pass

    def boundary_vertices(self):
        pass

    def boundary_edges(self):
        pass

    def get_edge(self, edge_id):
        edge_vertices = self.edge_vertices[edge_id]
        edge_coords = self.vertices[edge_vertices, :]

        return edge_coords

    def get_vertex(self, vertex_id):
        pass

    def get_edge_id(self, edge):
        pass

    def get_vertex_id(self, vertex):
        pass

    def incident_triangles(self, vertex_index):
        """
        return a list of the triangles that contain vertex v.
        :param vertex_index: vertex number
        :return: list of the triangles
        """

        triangle_idx = []
        for k in range(len(self.triangles)):
            if vertex_index in self.triangles[k]:
                triangle_idx.append(k)
        return triangle_idx

    def _generate_data(self):

        # unique edges in the mesh
        self.edge_vertices = np.array(list(set(tuple(sorted(edge))
                                               for triangle in self.triangles
                                               for edge in itertools.combinations(triangle, 2))))

        # edge to edge_index map - edges with reverse orientation map to same index
        self.edge_indices = {tuple(edge): i
                             for i, edge_ in enumerate(self.edge_vertices)
                             for edge in (edge_, reversed(edge_))}



if __name__ == '__main__':

    vertices = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    triangles = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ])

    M = Mesh(vertices, triangles)
