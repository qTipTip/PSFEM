import numpy as np
import quadpy
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import tqdm
from SSplines import ps12_sub_triangles


def quadpy_full(integrand, vertices, integration_order=20):
    return quadpy.triangle.integrate(integrand, vertices, scheme=quadpy.triangl.XiaoGimbutas(integration_order))


def quadpy_ps12(integrand, vertices, integration_order=20):
    integral = 0
    for sub_triangle in ps12_sub_triangles(vertices):
        integral += quadpy.triangle.integrate(integrand, sub_triangle,
                                              scheme=quadpy.triangle.XiaoGimbutas(integration_order))

    return integral


def interpolate_boundary_function(g, V):
    c = np.zeros(V.dimension)  # all coefficients

    bnd_val_dofs = [(v, V.value_dofs[v]) for v in V.value_dofs if V.value_dofs[v] in V.boundary_dofs]
    for vertex, dof in bnd_val_dofs:
        vertex_coordinates = V.mesh.get_vertex(vertex)
        c[dof] = g(vertex_coordinates)
    return c, bnd_val_dofs


def project_boundary_function(g, V):
    M = V.mesh
    A = sps.lil_matrix((V.dimension, V.dimension))
    c = np.zeros(V.dimension)
    coeffs = np.zeros(V.dimension)
    loc2glob = V.local_to_global_map

    def integrand(u, v):
        return lambda p: u(p.T) * v(p.T)

    def r_integrand(v):
        return lambda p: v(p.T) * g(p.T)

    visited_dofs = []
    for boundary_triangle in M.boundary_triangles():

        l2g = loc2glob[boundary_triangle]
        local_basis = [V.basis[basis_number].local_representation[boundary_triangle] for basis_number in l2g]
        for edge in M.get_boundary_edges(boundary_triangle):
            edge_vertices = M.vertices[list(edge)]
            for j in range(12):
                # skip functions not supported on the edge, outward normal derivatives for instance
                if l2g[j] not in V.boundary_dofs:
                    continue
                if l2g[j] in V.edge_dofs:
                    continue
                for i in range(j + 1):
                    if l2g[i] not in V.boundary_dofs:
                        continue
                    if l2g[i] in V.edge_dofs:
                        continue
                    I = np.sum(quadpy.nsimplex.integrate(integrand(local_basis[i], local_basis[j]), edge_vertices,
                                                         quadpy.nsimplex.GrundmannMoeller(1, 4)))
                    A[l2g[i], l2g[j]] += I

                    if i != j:
                        A[l2g[j], l2g[i]] += I
                visited_dofs.append(l2g[j])
                c[l2g[j]] += np.sum(quadpy.nsimplex.integrate(r_integrand(local_basis[j]), edge_vertices,
                                                              quadpy.nsimplex.GrundmannMoeller(1, 4)))
    A = sps.csr_matrix(A)
    c, _info = spla.gmres(A, c)  # TODO: This does not work with a direct solver for some reason.

    return c


def solve(a, L, V, verbose=False, nprocs=1, aorder=2, border=2, dirichlet=None):
    """
    Solves the discrete finite element problem
    Find u in V such that
        a(u, v) = L(v)
    for all v in V.

    :param a: bilinear form
    :param L: linear form
    :param V: Composite C^1 function space.
    :return: CompositeSpline u satisfying a(u, v) = L(v) for all v in V.
    """

    A = sps.lil_matrix((V.dimension, V.dimension))
    b = np.zeros(V.dimension)

    if dirichlet is None:
        dirichlet = lambda x: 0

    u_boundary_coeff = project_boundary_function(dirichlet, V)

    def compute_single_triangle(triangle):
        triangle_coords = V.mesh.vertices[V.mesh.triangles[triangle]]
        l2g = V.local_to_global_map[triangle]
        local_basis = [V.basis[basis_number].local_representation[triangle] for basis_number in l2g]
        for j in tqdm.trange(12, leave=False, disable=not verbose, desc='   Local assembly'):
            for i in range(j + 1):
                I = quadpy_ps12(a(local_basis[i], local_basis[j]), triangle_coords, integration_order=aorder)
                A[l2g[i], l2g[j]] += I
                if i != j:
                    A[l2g[j], l2g[i]] += I

            b[l2g[j]] += quadpy_ps12(L(local_basis[j]), triangle_coords, integration_order=border)
    for triangle in tqdm.trange(len(V.mesh.triangles), disable=not verbose, desc='Global assembly'):
        compute_single_triangle(triangle)

    interior_dofs = V.interior_dofs
    boundary_dofs = V.boundary_dofs

    # modify matrix and vector to incorporate boundary conditions
    for boundary_dof in boundary_dofs:
        A[boundary_dof, :] = np.zeros(V.dimension)
        A[boundary_dof, boundary_dof] = 1
        b[boundary_dof] = u_boundary_coeff[boundary_dof]

    A = sps.csr_matrix(A)
    c = spla.spsolve(A, b)

    return c
