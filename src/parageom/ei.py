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


def interpolate_subdomain_operator(operator):
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.algorithms.ei import deim

    parameter_space = operator.parameters.space(example.mu_range)
    training_set = parameter_space.sample_uniformly(101)

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
    interpolation_dofs, collateral_basis, deim_data = deim(Λ, rtol=1e-6)

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
    cb, interpmat, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(operator)
    r_op, source_dofs = operator.restricted(magic_dofs)

    test_mu = operator.parameters.parse([287.])
    # ### Reference matrix
    kref = csr_array(operator.assemble(test_mu).matrix.getValuesCSR()[::-1])

    # ### compare DEIM approximation
    AU = csr_array(r_op.assemble(test_mu).matrix.getValuesCSR()[::-1])
    AU_dofs = AU[r_op.restricted_range_dofs, r_op.restricted_range_dofs]
    interpolation_coefficients = solve(interpmat, AU_dofs)

    ei_approx = LincombOperator(cb, interpolation_coefficients)
    K = ei_approx.assemble().matrix

    abserr = norm(kref - K, ord="fro")
    relerr = norm(kref - K, ord="fro") / norm(kref, ord="fro")
    print(f"{abserr=} of MDEIM approximation in Frobenious norm.")
    print(f"{relerr=} of MDEIM approximation in Frobenious norm.")

    interp_mat_inv = np.linalg.inv(interpmat)
    # FIXME: actually need to get the first discarded singular value
    sigma_m_1 = deim_data['svals'][-1]
    bound = np.linalg.norm(interp_mat_inv) * sigma_m_1
