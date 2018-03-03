import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import tqdm

from SSplines import gaussian_quadrature_data, gaussian_quadrature_ps12


def interpolate_boundary_function(g, V):
    c = np.zeros(V.dimension)  # all coefficients

    bnd_val_dofs = [(v, V.value_dofs[v]) for v in V.value_dofs if V.value_dofs[v] in V.boundary_dofs]
    for vertex, dof in bnd_val_dofs:
        vertex_coordinates = V.mesh.get_vertex(vertex)
        c[dof] = g(vertex_coordinates)
    return c


def solve(a, L, V, verbose=False, nprocs=1, integration_method=gaussian_quadrature_ps12, order=2, dirichlet=None):
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

    if dirichlet is None:
        dirichlet = lambda p: 0

    u_boundary_coefficients = interpolate_boundary_function(dirichlet, V)
    u_boundary = V.function(u_boundary_coefficients)
    A = sps.lil_matrix((V.dimension, V.dimension))
    b = np.zeros(V.dimension)
    c = np.zeros(V.dimension)

    quad_points, quad_weights = gaussian_quadrature_data(order)

    def compute_single_triangle(triangle):
        triangle_coords = V.mesh.vertices[V.mesh.triangles[triangle]]
        l2g = V.local_to_global_map[triangle]
        local_basis = [V.basis[basis_number].local_representation[triangle] for basis_number in l2g]
        for j in tqdm.trange(12, leave=False, disable=not verbose, desc='   Local assembly'):
            for i in range(j + 1):
                I = integration_method(triangle_coords, a(local_basis[i], local_basis[j]), quad_points, quad_weights)
                A[l2g[i], l2g[j]] += I
                if i != j:
                    A[l2g[j], l2g[i]] += I

            b[l2g[j]] += integration_method(triangle_coords, L(local_basis[j]), quad_points,
                                            quad_weights) - integration_method(triangle_coords,
                                                                               a(u_boundary, local_basis[j]),
                                                                               quad_points,
                                                                               quad_weights)  # add correctiont erm

    for triangle in tqdm.trange(len(V.mesh.triangles), disable=not verbose, desc='Global assembly'):
        compute_single_triangle(triangle)

    # interior_dofs = V.interior_dofs
    A = sps.csr_matrix(A)
    # c[interior_dofs] = spla.spsolve(A[np.ix_(interior_dofs, interior_dofs)], b[interior_dofs])
    c = spla.spsolve(A, b)

    return c + u_boundary_coefficients
