from collections import defaultdict
from mpi4py import MPI
from dolfinx import mesh
import numpy as np

from multi.bcs import apply_bcs
from multi.dofmap import DofMap
from multi.domain import StructuredQuadGrid
from multi.boundary import plane_at
from .locmor import COOMatrixOperator

from pymor.parameters.base import Parameters, ParameterSpace
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator

elementmatrix = np.array([
    [1., 0.5, 0.5, 0.25],
    [0.5, 1., 0.25, 0.5],
    [0.5, 0.25, 1., 0.5],
    [0.25, 0.5, 0.5, 1.]
    ])

elementvector = np.array([
    [1.], [2.], [2.], [1.]
    ])


def test():
    """homogeneous dirichlet bcs"""
    nx = ny = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
    quad_grid = StructuredQuadGrid(domain)
    dofmap = DofMap(quad_grid)
    dofmap.distribute_dofs(1, 0)
    N = dofmap.num_dofs
    num_cells = quad_grid.num_cells

    # ### Boundary condition
    bottom = quad_grid.locate_entities_boundary(0, plane_at(0., "y"))
    bottom_dofs = []
    for vertex in bottom:
        bottom_dofs += dofmap.entity_dofs(0, vertex)
    bc_dofs = np.array(bottom_dofs)
    bc_vals = np.zeros_like(bc_dofs)

    # ### Parameter space
    P = ParameterSpace(Parameters({"mu": num_cells}), (0., 5.))
    test_mu = P.sample_randomly(1)[0]

    # ### Assemble parametric A and apply BCs globally
    A = np.zeros((N, N))
    mu = test_mu.to_numpy()
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        A[np.ix_(dofs, dofs)] += mu[ci] * elementmatrix
    b = np.zeros((N,))
    apply_bcs(A, b, bc_dofs, bc_vals)

    # ### Use COO matrix, but apply BCs locally
    diagonals = []
    data = []
    rows = []
    cols = []
    indexptr = []
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        for l, x in enumerate(dofs):
            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:

                    # Note: in the MOR context set diagonal to zero
                    # for the matrices arising from a_q
                    if x == y:
                        if x not in diagonals: # only set diagonal entry once
                            rows.append(x)
                            cols.append(y)
                            data.append(0.0)
                            diagonals.append(x)
                        
                else:
                    rows.append(x)
                    cols.append(y)
                    data.append(elementmatrix[l, k])
        indexptr.append(len(rows))

    options = None
    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)
    indexptr = np.array(indexptr)
    shape = (N, N)
    op = COOMatrixOperator((data, rows, cols), indexptr, num_cells, shape, solver_options=options, name="K")

    K = op.assemble(test_mu)
    assert K.name == "K_assembled"
    Karray = K.matrix.toarray()
    K_bc = np.zeros_like(Karray)
    apply_bcs(K_bc, b, bc_dofs, bc_vals)
    assert np.allclose(A, K_bc + Karray)


def test_rhs():
    """homogeneous dirichlet bcs"""
    nx = ny = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
    quad_grid = StructuredQuadGrid(domain)
    dofmap = DofMap(quad_grid)
    dofmap.distribute_dofs(1, 0)
    N = dofmap.num_dofs
    num_cells = quad_grid.num_cells

    # ### Boundary condition
    bottom = quad_grid.locate_entities_boundary(0, plane_at(0., "y"))
    bottom_dofs = []
    for vertex in bottom:
        bottom_dofs += dofmap.entity_dofs(0, vertex)
    bc_dofs = np.array(bottom_dofs)
    bc_vals = np.zeros_like(bc_dofs)

    # ### Parameter space
    P = ParameterSpace(Parameters({"mu": num_cells}), (0., 5.))
    test_mu = P.sample_randomly(1)[0]

    # ### Assemble parametric rhs and apply BCs globally
    A = np.zeros((N, N))
    F = np.zeros((N, 1))
    mu = test_mu.to_numpy()
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        A[np.ix_(dofs, dofs)] += mu[ci] * elementmatrix
        F[dofs, :] += mu[ci] * elementvector

    apply_bcs(A, F, bc_dofs, bc_vals)

    # ### Use COO matrix for LHS, but apply BCs locally
    lhs = defaultdict(list)
    rhs = defaultdict(list)

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        for l, x in enumerate(dofs):

            if x in bc_dofs:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(0.0)
            else:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(elementvector[l, 0])

            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:
                    # Note: in the MOR context set diagonal to zero
                    # for the matrices arising from a_q
                    if x == y:
                        if x not in lhs["diagonals"]: # only set diagonal entry once
                            lhs["rows"].append(x)
                            lhs["cols"].append(y)
                            lhs["data"].append(0.0)
                            lhs["diagonals"].append(x)
                else:
                    lhs["rows"].append(x)
                    lhs["cols"].append(y)
                    lhs["data"].append(elementmatrix[l, k])


        lhs["indexptr"].append(len(lhs["rows"]))
        rhs["indexptr"].append(len(rhs["rows"]))

    data = np.array(lhs["data"])
    rows = np.array(lhs["rows"])
    cols = np.array(lhs["cols"])
    indexptr = np.array(lhs["indexptr"])
    shape = (N, N)
    parameters = {"mu": 4}
    options = None
    op = COOMatrixOperator((data, rows, cols), indexptr, num_cells, shape, parameters=parameters, solver_options=options, name="K")
    assert op.parametric
    # op.parametric returns False, but should be True for correct assembly I guess
    # also LincombOperator.assemble modifies the name of the operators
    # I should probably not mess with the name?

    K = op.assemble(test_mu)
    assert K.name == "K_assembled"
    Karray = K.matrix.toarray()
    K_bc = np.zeros_like(Karray)
    b = np.zeros(N)
    apply_bcs(K_bc, b, bc_dofs, bc_vals)
    assert np.allclose(A, K_bc + Karray)

    bc_op = NumpyMatrixOperator(K_bc)
    lincombop = LincombOperator([op, bc_op], [1., 1.])
    L = lincombop.assemble(test_mu)
    assert np.allclose(A, L.matrix)

    options = None
    data = np.array(rhs["data"])
    rows = np.array(rhs["rows"])
    cols = np.array(rhs["cols"])
    indexptr = np.array(rhs["indexptr"])
    shape = (N, 1)
    rhs_op = COOMatrixOperator((data, rows, cols), indexptr, num_cells, shape, solver_options=options, name="F")
    assert not rhs_op.parametric
    rhs = rhs_op.assemble(test_mu)
    other = rhs_op.assemble()
    assert np.allclose(F.reshape(9, 1), rhs.matrix.todense())
    assert np.allclose(F.reshape(9, 1), other.matrix.todense())


if __name__ == "__main__":
    test()
    test_rhs()
