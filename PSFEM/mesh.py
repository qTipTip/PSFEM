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