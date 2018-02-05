import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import tqdm

from PSFEM.quadrature import midpoint_rule_ps12


def solve(a, L, V, verbose=False):
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

    for triangle in tqdm.trange(len(V.mesh.triangles), disable=not verbose):
        triangle_coords = V.mesh.vertices[V.mesh.triangles[triangle]]
        l2g = V.local_to_global_map[triangle]
        local_basis = V.local_spline_bases[triangle]
        for j in tqdm.trange(12, leave=False, disable=not verbose):
            for i in range(j + 1):
                I = midpoint_rule_ps12(a(local_basis[i], local_basis[j]), triangle_coords)
                A[l2g[i], l2g[j]] += I
                if i != j:
                    A[l2g[j], l2g[i]] += I
            b[l2g[j]] += midpoint_rule_ps12(L(local_basis[j]), triangle_coords)

    interior_dofs = V.interior_dofs
    A = sps.csr_matrix(A)
    c[interior_dofs] = spla.spsolve(A[np.ix_(interior_dofs, interior_dofs)], b[interior_dofs])

    return V.function(c)