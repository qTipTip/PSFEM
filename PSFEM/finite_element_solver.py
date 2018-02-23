import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import tqdm

from SSplines import gaussian_quadrature_data, gaussian_quadrature_ps12

quad_points, quad_weights = gaussian_quadrature_data(2)


def solve(a, L, V, verbose=False, nprocs=1):
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
    c = np.zeros(V.dimension)

    B, W = gaussian_quadrature_data(2)

    def compute_single_triangle(triangle):
        triangle_coords = V.mesh.vertices[V.mesh.triangles[triangle]]
        l2g = V.local_to_global_map[triangle]
        local_basis = [V.basis[basis_number].local_representation[triangle] for basis_number in l2g]
        for j in tqdm.trange(12, leave=False, disable=not verbose, desc='   Local assembly'):
            for i in range(j + 1):
                I = gaussian_quadrature_ps12(triangle_coords, a(local_basis[i], local_basis[j]), B, W)
                A[l2g[i], l2g[j]] += I
                if i != j:
                    A[l2g[j], l2g[i]] += I

            b[l2g[j]] += gaussian_quadrature_ps12(triangle_coords, (local_basis[j]), B, W)

    for triangle in tqdm.trange(len(V.mesh.triangles), disable=not verbose, desc='Global assembly'):
        compute_single_triangle(triangle)

    interior_dofs = V.interior_dofs
    A = sps.csr_matrix(A)
    c[interior_dofs] = spla.spsolve(A[np.ix_(interior_dofs, interior_dofs)], b[interior_dofs])

    return V.function(c)
