"""empirical interpolation module"""

from typing import Optional

import numpy as np
from scipy.sparse import csr_array


def vec(petsc_mat):
    return petsc_mat.getValuesCSR()[2]

def vec2mat(A: csr_array):
    # map from j = 2r + c to row and col
    rowscols = []
    nrows, _ = A.shape # A sparse matrix
    for r in range(nrows):
        cols = A.indices[A.indptr[r]:A.indptr[r+1]]
        for c in cols:
            rowscols.append((r, c))
    mapping = np.vstack(rowscols) # maps from j = 0, ..., nz-1 to (row, col)
    return mapping


def interpolate_subdomain_operator(example, operator, design: str="lhs", ntrain: int=101, modes: Optional[int] = None, atol: Optional[float] = None, rtol: Optional[float] = None, method: Optional[str] = "method_of_snapshots"):
    """EI of subdomain operator.

    Args:
        example: data class.
        operator: FenicsxMatrixBasedOperator to interpolate.
        design: Design for the sampling of training set (choices: uniform, random, lhs).
        ntrain: Number of training samples for the DEIM.
        modes: Use at most `modes` POD modes with DEIM.
        atol: Absolute tolerance for the POD with DEIM.
        rtol: Relative tolerance for the POD with DEIM.
        method: POD method (choices: method_of_snapshots, qr_svd).
    """
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.algorithms.ei import deim
    from .lhs import sample_lhs

    parameter_space = operator.parameters.space(example.mu_range)
    parameter_name = list(example.parameters["subdomain"].keys())[0]
    if design == "uniform":
        training_set = parameter_space.sample_uniformly(ntrain)
    elif design == "random":
        training_set = parameter_space.sample_randomly(ntrain)
    elif design == "lhs":
        training_set = sample_lhs(
                parameter_space,
                name=parameter_name,
                samples=ntrain,
                criterion="center",
                random_state=25525298)
    else:
        raise NotImplementedError

    # ### build map
    mu_0 = training_set[0]
    K = csr_array(operator.assemble(mu=mu_0).matrix.getValuesCSR()[::-1])
    index_map = vec2mat(K)
    
    # ### snapshots
    vec_source = NumpyVectorSpace(K.nnz)
    snapshots = [K.data, ]
    for mu in training_set[1:]:
        matop = operator.assemble(mu)
        snapshots.append(vec(matop.matrix))
    Λ = vec_source.make_array(snapshots)

    # ### DEIM
    pod_options = {"method": method}
    interpolation_dofs, collateral_basis, deim_data = deim(Λ, modes=modes, pod=True, atol=atol, rtol=rtol, product=None, pod_options=pod_options)

    # ### outputs/targets
    # reorder idofs and collateral basis
    # this is necessary because the dofs are ordered in ascending order
    # when the dof mapping from full mesh to the submesh space is computed
    ind = np.argsort(interpolation_dofs)
    idofs = interpolation_dofs[ind]
    collateral_basis = collateral_basis[ind]

    # interpolation dofs in the matrix format
    magic_dofs = index_map[idofs]
    interpolation_matrix = collateral_basis.dofs(idofs).T

    # FIXME
    # index_map[idofs] maps the index j in the vector format to a pair of row & col indices
    # the user has to do map only unique magic dofs for evaluation in the restricted range
    # (i.e. on the submesh) and use the `return_inverse` option with np.unique, to be able
    # to reconstruct pairs of row & col indices in the restricted range.
    # See __main__ below.

    # ### matrix operators (reverse vec operation)
    # collateral matrices
    mops = []
    indptr = K.indptr
    indices = K.indices
    for i in range(len(collateral_basis)):
        data = collateral_basis[i].to_numpy().flatten()
        cbm = NumpyMatrixOperator(csr_array((data, indices, indptr), shape=K.shape))
        mops.append(cbm)

    return mops, interpolation_matrix, idofs, magic_dofs, deim_data


if __name__ == "__main__":
    from .tasks import example
    from .fom import discretize_subdomain_operators
    from scipy.linalg import solve
    from scipy.sparse.linalg import norm
    from pymor.operators.constructions import LincombOperator

    operator, _ = discretize_subdomain_operators(example)
    cb, interpmat, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator, design="uniform", ntrain=501, modes=None, atol=0., rtol=1e-12)
    m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
    r_op, source_dofs = operator.restricted(m_dofs)
    range_dofs = r_op.restricted_range_dofs[m_inv].reshape(magic_dofs.shape)

    pspace = operator.parameters.space((0.1, 0.3))
    test_set = pspace.sample_randomly(50)

    abserr = []
    relerr = []

    for mu in test_set:
        # ### compare DEIM approximation
        # Note, need to get the matrix instead of vectorized entries, because
        # r_op.restricted_range_dofs points to dofs of V
        AU = csr_array(r_op.assemble(mu).matrix.getValuesCSR()[::-1])
        AU_dofs = AU[range_dofs[:, 0], range_dofs[:, 1]]
        try:
            interpolation_coefficients = solve(interpmat, AU_dofs)
        except ValueError:
            breakpoint()

        ei_approx = LincombOperator(cb, interpolation_coefficients)
        K = ei_approx.assemble().matrix

        # ### Reference matrix
        kref = csr_array(operator.assemble(mu).matrix.getValuesCSR()[::-1])

        abserr.append(norm(kref - K, ord="fro"))
        relerr.append(norm(kref - K, ord="fro") / norm(kref, ord="fro"))

    print(f"Max absolute error in Frobenious norm:\t{np.max(abserr)}")
    print(f"Max relative error in Frobenious norm:\t{np.max(relerr)}")
