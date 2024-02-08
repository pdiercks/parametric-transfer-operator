from mpi4py import MPI
from dolfinx import mesh
import numpy as np

from multi.bcs import apply_bcs
from multi.dofmap import DofMap
from multi.domain import StructuredQuadGrid
from multi.boundary import plane_at
from .locmor import COOMatrixOperator

from pymor.parameters.base import Parameters, ParameterSpace

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
    b = np.zeros(N)
    apply_bcs(K_bc, b, bc_dofs, bc_vals)
    assert np.allclose(A, K_bc + Karray)

    # ### Use COO matrix for RHS, but apply BCs locally
    data = []
    rows = []
    cols = []
    indexptr = []
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        for l, x in enumerate(dofs):
            y = 0
            if x in bc_dofs:
                rows.append(x)
                cols.append(y)
                data.append(0.0)
                    
            else:
                rows.append(x)
                cols.append(y)
                data.append(elementvector[l, 0])
        indexptr.append(len(rows))

    options = None
    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)
    indexptr = np.array(indexptr)
    shape = (N, 1)
    op = COOMatrixOperator((data, rows, cols), indexptr, num_cells, shape, solver_options=options, name="F")
    rhs = op.assemble(test_mu)
    assert np.allclose(F.reshape(9, 1), rhs.matrix.todense())


if __name__ == "__main__":
    test()
    test_rhs()
