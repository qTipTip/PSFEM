import numpy as np
from SSplines import SplineSpace

from PSFEM.helper_functions import local_to_global


class CompositeSpline(object):
    def __init__(self, local_representation, triangles_with_support, mesh):
        self.local_representation = local_representation
        self.triangles_with_support = triangles_with_support
        self.mesh = mesh
        self.last_triangle = 0  # last triangle evaluated in

    def __call__(self, x, k=None):

        if k is None:
            k = self.mesh.find_triangle(x, hint=self.last_triangle)
        # if x lies in a supported triangle, evaluate
        self.last_triangle = k
        if k in self.triangles_with_support:
            return self.local_representation[k](x)
        # else, return 0
        else:
            return np.zeros(len(x))

    def __mul__(self, scalar):
        new_local_representation = {}
        for k in self.triangles_with_support:
            new_local_representation[k] = self.local_representation[k] * scalar
        return CompositeSpline(new_local_representation, self.triangles_with_support, self.mesh)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, other):
        new_triangles_with_support = list(set(self.triangles_with_support + other.triangles_with_support))
        new_local_representation = {}

        for k in self.triangles_with_support:
            new_local_representation[k] = self.local_representation[k]
        for k in other.triangles_with_support:
            if k in new_local_representation.keys():
                new_local_representation[k] += other.local_representation[k]
            else:
                new_local_representation[k] = other.local_representation[k]

        return CompositeSpline(new_local_representation, new_triangles_with_support, self.mesh)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def lapl(self, x, k=None):
        # if x lies in a supported triangle, evaluate
        if k is None:
            k = self.mesh.find_triangle(x, hint=self.last_triangle)
        if k in self.triangles_with_support:
            return self.local_representation[k].lapl(x)
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
        self.local_to_global_map, self.dof_to_edge_map, self.dof_to_vertex_map, self.value_dofs, self.dx_dofs, \
        self.dy_dofs, self.edge_dofs = local_to_global(mesh.vertices, mesh.triangles)
        self.local_spline_spaces = [SplineSpace(mesh.vertices[triangle], degree=2) for triangle in mesh.triangles]
        self.local_spline_bases = [S.hermite_basis() for S in self.local_spline_spaces]
        self._construct_basis_to_triangle_map()
        self._construct_global_to_local_map()
        self._construct_interior_and_boundary_dofs()
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

    def _construct_interior_and_boundary_dofs(self):
        interior_dofs = []
        boundary_dofs = []

        for dof in self.dof_to_edge_map.keys():
            edge_id = self.mesh.get_edge_id(self.dof_to_edge_map[dof])
            if edge_id in self.mesh.bnd_edges:
                boundary_dofs.append(dof)
            else:
                interior_dofs.append(dof)

        for dof in self.dof_to_vertex_map.keys():
            if self.dof_to_vertex_map[dof] in self.mesh.bnd_vertices:
                boundary_dofs.append(dof)
            else:
                interior_dofs.append(dof)

        self.interior_dofs = sorted(interior_dofs)
        self.boundary_dofs = sorted(boundary_dofs)

    def function(self, coefficients):
        """
        Returns a callable CompositeSpline function.
        :param coefficients:
        :return:
        """
        return sum([c*b for c, b in zip(coefficients, self.basis)])
