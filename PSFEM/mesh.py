import itertools

import numpy as np
from SSplines.helper_functions import barycentric_coordinates


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
        self._compute_h()

    def interior_vertices(self):
        return self.int_vertices

    def interior_edges(self):
        return self.int_edges

    def boundary_vertices(self):
        return self.bnd_vertices

    def boundary_edges(self):
        return self.bnd_edges

    def get_edge(self, edge_id):
        edge_vertices = self.edge_vertices[edge_id]
        edge_coords = self.vertices[edge_vertices, :]

        return edge_coords

    def get_vertex(self, vertex_id):
        return self.vertices[vertex_id]

    def get_edge_id(self, edge):
        return self.edge_indices[edge]

    def get_vertex_id(self, vertex):
        for k in range(len(self.vertices)):
            if vertex == self.vertices[k]:
                return k
        return -1

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

    def adjacent_triangles(self, edge_index):
        """
        Return a list of the triangles sharing edge edge_index
        :param edge_index: edge index
        :return: list of triangles sharing edge.
        """

        edge = self.edge_vertices[edge_index]
        triangle_idx = []
        for k in range(len(self.triangles)):
            if edge[0] in self.triangles[k] and edge[1] in self.triangles[k]:
                triangle_idx.append(k)
        return triangle_idx

    def _compute_h(self):
        """
        Computes the max, min and average triangle side-length in the mesh.
        """

        edge_lengths = np.zeros(len(self.edges))

        for e in self.edges:
            i, j = self.edge_vertices[e]
            v1, v2 = self.vertices[[i, j], :]
            edge_lengths[e] = np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)

        self.h_max = max(edge_lengths)
        self.h_min = min(edge_lengths)
        self.h_avg = np.average(edge_lengths)

    def _generate_data(self):

        # unique edges in the mesh
        self.edge_vertices = np.array(list(set(tuple(sorted(edge))
                                               for triangle in self.triangles
                                               for edge in itertools.combinations(triangle, 2))))

        # edge to edge_index map - edges with reverse orientation map to same index
        self.edge_indices = {tuple(edge): i
                             for i, edge_ in enumerate(self.edge_vertices)
                             for edge in (edge_, reversed(edge_))}

        # list of boundary edges
        self.bnd_edges = sorted([self.edge_indices[tuple(edge)] for edge in self.edge_vertices
                          if len(self.adjacent_triangles(self.edge_indices[tuple(edge)])) == 1])

        # list of boundary vertices
        self.bnd_vertices = sorted(list(set([vertex
                             for edge in self.bnd_edges
                             for vertex in self.edge_vertices[edge]])))

        self.int_edges = [self.edge_indices[tuple(edge)] for edge in self.edge_vertices
                          if self.edge_indices[tuple(edge)] not in self.bnd_edges]

        self.int_vertices = [vertex_id for vertex_id in range(len(self.vertices)) if vertex_id not in self.bnd_vertices]

        self.edges = self.bnd_edges + self.int_edges

    def find_triangle(self, x, hint=None):
        """
        Given a point x in the domain of the triangulation, determine for which index i
        the point x lies in triangle i.
        :param np.ndarray x: point of interest
        :return: index i such that x lies in T_i
        """

        # see if the point lies within the hinted triangle.
        if hint is not None:
            hint_vertices = self.vertices[self.triangles[hint]]
            b = barycentric_coordinates(hint_vertices, x)

            if np.all(b >= 0):
                return hint

        for k, T in enumerate(self.triangles):
            vertices = self.vertices[T]
            b = barycentric_coordinates(vertices, x)

            if np.all(b >= 0):
                return k
            else:
                continue

        return k
