"""empirical interpolation module"""

import dolfinx as df
import numpy as np
from scipy.sparse import csr_array


def vec(petsc_mat):
    return petsc_mat.getValuesCSR()[2]

def vec2mat(A):
    # map from j = 2r + c to row and col
    rowscols = []
    nrows, _ = A.shape # A sparse matrix
    for r in range(nrows):
        cols = A.indices[A.indptr[r]:A.indptr[r+1]]
        for c in cols:
            rowscols.append((r, c))
    mapping = np.vstack(rowscols) # maps from j = 0, ..., nz-1 to (row, col)
    return mapping


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .matrix_based_operator import FenicsxMatrixBasedOperator
    from .fom import ParaGeomLinEla
    from multi.domain import RectangularDomain
    from pymor.vectorarrays.numpy import NumpyVectorSpace
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.algorithms.ei import deim

    # ### Discretize operator
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, ftags, example.parameters["subdomain"]
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    EMOD = example.youngs_modulus
    POISSON = example.poisson_ratio
    domain = aux.problem.domain.grid
    omega = RectangularDomain(domain)
    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d)

    # ### wrap as pymor model
    def param_setter(mu):
        d.x.array[:] = 0.
        aux.solve(d, mu)
        d.x.scatter_forward()

    params = {"R": 1}
    operator = FenicsxMatrixBasedOperator(problem.form_lhs, params, param_setter=param_setter, name="ParaGeom")

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
    interpolation_dofs, collateral_basis, deim_data = deim(Λ, rtol=1e-5)

    # ### outputs/targets
    # interpolation dofs in the matrix format
    magic_dofs = np.unique(index_map[interpolation_dofs])
    interpolation_matrix = collateral_basis.dofs(interpolation_dofs).T

    # ### matrix operators (reverse vec operation)
    mops = []
    indptr = K.indptr
    indices = K.indices
    for i in range(len(collateral_basis)):
        data = collateral_basis[i].to_numpy().flatten()
        cbm = NumpyMatrixOperator(csr_array((data, indices, indptr), shape=K.shape))
        mops.append(cbm)

    # TODO
    # make the above a function

    # use this function in the script that builds the global ROM
    # and write the operators to disk
    # see src/beam/run_locrom.py

    # Online: mu (size 10) --> solve interpolation equation for each subdomain individually
    # each interpolation eq. yields coefficient vector of length M
    # form linear combination of local operators for each subdomain
    # then assemble global operators and solve

    # Using COOMatrixOperator has advantage that it fits better in pymor interfaces
    # Would have 1 COOMatrixOperator per collateral basis function
    # the corresponding parameter functional would be the i-th entry of each of the
    # local interpolation coefficient vectors

    # args
    # realization index
    # method (hapod, heuristic)
    # distribution
    # number of modes per edge


if __name__ == "__main__":
    main()
