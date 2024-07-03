"""empirical interpolation module"""

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


def interpolate_subdomain_operator(example, operator, design: str="lhs", ntrain: int=101, rtol: float=1e-5):
    """EI of subdomain operator.

    Args:
        example: data class.
        operator: FenicsxMatrixBasedOperator to interpolate.
        ntrain: Number of training samples for the DEIM.
        rtol: Relative tolerance for the POD with DEIM.
    """
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.algorithms.ei import deim
    from .lhs import sample_lhs

    parameter_space = operator.parameters.space(example.mu_range)
    parameter_name = list(example.parameters["subdomain"].keys())[0]
    if design == "uniform":
        training_set = parameter_space.sample_uniformly(ntrain)
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
    interpolation_dofs, collateral_basis, deim_data = deim(Λ, rtol=rtol)

    # ### outputs/targets

    # reorder idofs and collateral basis
    ind = np.argsort(interpolation_dofs)
    idofs = interpolation_dofs[ind]
    collateral_basis = collateral_basis[ind]

    # interpolation dofs in the matrix format
    magic_dofs = np.unique(index_map[idofs])
    interpolation_matrix = collateral_basis.dofs(idofs).T

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
    cb, interpmat, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator)
    r_op, source_dofs = operator.restricted(magic_dofs)

    pspace = operator.parameters.space((0.1, 0.3))
    test_set = pspace.sample_randomly(30)

    abserr = []
    relerr = []

    for mu in test_set:
        # ### Reference matrix
        kref = csr_array(operator.assemble(mu).matrix.getValuesCSR()[::-1])

        # ### compare DEIM approximation
        # Note, need to get the matrix instead of vectorized entries, because
        # r_op.restricted_range_dofs points to dofs of V
        AU = csr_array(r_op.assemble(mu).matrix.getValuesCSR()[::-1])
        AU_dofs = AU[r_op.restricted_range_dofs, r_op.restricted_range_dofs]
        interpolation_coefficients = solve(interpmat, AU_dofs)

        ei_approx = LincombOperator(cb, interpolation_coefficients)
        K = ei_approx.assemble().matrix

        abserr.append(norm(kref - K, ord="fro"))
        relerr.append(norm(kref - K, ord="fro") / norm(kref, ord="fro"))

    print(f"Max absolute error in Frobenious norm:\t{np.max(abserr)}")
    print(f"Max relative error in Frobenious norm:\t{np.max(relerr)}")
