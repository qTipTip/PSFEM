import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from SSplines import SplineSpace, sample_triangle

from PSFEM.helper_functions import local_to_global
from PSFEM.mesh import Mesh


class CompositeSpline(object):

    def __init__(self, local_representation, triangles_with_support, mesh):
        self.mesh = mesh
        self.local_representation = local_representation
        self.triangles_with_support = triangles_with_support

    def __call__(self, x, k):
        # if x lies in a supported triangle, evaluate
        if k in self.triangles_with_support:
            return self.local_representation[k](x)
        # else, return 0
        else:
            return np.zeros(len(x))


class CompositeSplineSpace(object):

    def __init__(self, mesh):
        """
        Initializes a Composite C^1 spline space over the given mesh.
        :param mesh:
        """

        self.mesh = mesh
        self.dimension = 3*len(mesh.vertices) + len(mesh.edges)
        self.local_to_global_map, self.dof_to_edge_map, self.dof_to_vertex_map = local_to_global(mesh.vertices, mesh.triangles)
        self.local_spline_spaces = [SplineSpace(mesh.vertices[triangle], degree=2) for triangle in mesh.triangles]
        self._construct_basis_to_triangle_map()
        self._construct_global_to_local_map()
        self.basis = [self._construct_global_basis_function(i) for i in range(self.dimension)]

    def _construct_global_to_local_map(self):
        """
        Constructs a map mapping a dof index to a dictionary that maps triangle of support to local representation
        on that triangle.
        :return: dict
        """

        global_to_local_map = {}

        for dof in range(self.dimension):
            triangles_of_support = self.basis_to_triangle_map[dof]
            global_to_local_map[dof] = {triangle: self.local_to_global_map[triangle].index(dof) for triangle in triangles_of_support}

        self.global_to_local_map = global_to_local_map

    def _construct_basis_to_triangle_map(self):

        basis_to_triangle = {}

        for dof in range(self.dimension):
            # if dof is an edge dof
            if dof in self.dof_to_edge_map.keys():
                edge_vertices = self.dof_to_edge_map[dof]
                edge_index = self.mesh.edge_indices[edge_vertices]
                basis_to_triangle[dof] = self.mesh.adjacent_triangles(edge_index)
            # otherwise, its a vertex dof
            else:
                vertex_index = self.dof_to_vertex_map[dof]
                basis_to_triangle[dof] = self.mesh.incident_triangles(vertex_index)

        self.basis_to_triangle_map = basis_to_triangle

    def _construct_global_basis_function(self, i):

        triangles_with_support = self.basis_to_triangle_map[i]
        local_representation = {}
        for triangle in triangles_with_support:
            s = self.local_spline_spaces[triangle]
            b = s.hermite_basis()[self.global_to_local_map[i][triangle]]
            local_representation[triangle] = b

        # if dof is an interior edge dof, we flip the sign on one of the two triangles
        # to enforce C^1 smoothness across the edge.
        if i in self.dof_to_edge_map.keys():
            edge_vertex = self.dof_to_edge_map[i]
            edge_idx = self.mesh.edge_indices[edge_vertex]
            if edge_idx in self.mesh.int_edges:
                local_representation[triangles_with_support[1]] *= -1

        return CompositeSpline(local_representation, triangles_with_support, self.mesh)

if __name__ == '__main__':

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
    C = CompositeSplineSpace(M)

    points = [sample_triangle(vertices[triangles[k]], 10) for k in range(len(triangles))]

    for basis in C.basis:
        fig = plt.figure()
        axs = Axes3D(fig)
        axs.set_zlim3d(-0.1, 1)
        for k in range(len(triangles)):
            p = points[k]
            z = basis(p, k)
            axs.plot_trisurf(p[:, 0], p[:, 1], z)
        plt.show()
