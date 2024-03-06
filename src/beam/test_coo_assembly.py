from mpi4py import MPI
from dolfinx import mesh
# from dolfinx.fem.petsc import assemble_matrix
# from basix.ufl import element
# import ufl
from scipy.sparse import coo_array
import numpy as np

from multi.bcs import apply_bcs
from multi.boundary import plane_at
from multi.domain import StructuredQuadGrid
from multi.dofmap import DofMap


elementmatrix = np.array([
    [1., 0.5, 0.5, 0.25],
    [0.5, 1., 0.25, 0.5],
    [0.5, 0.25, 1., 0.5],
    [0.25, 0.5, 0.5, 1.]
    ])


def test():
    """homogeneous dirichlet bcs"""
    nx = ny = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.quadrilateral)
    quad_grid = StructuredQuadGrid(domain)
    dofmap = DofMap(quad_grid)
    dofmap.distribute_dofs(1, 0)
    N = dofmap.num_dofs

    # ### Boundary condition
    bottom = quad_grid.locate_entities_boundary(0, plane_at(0., "y"))
    bottom_dofs = []
    for vertex in bottom:
        bottom_dofs += dofmap.entity_dofs(0, vertex)
    bc_dofs = np.array(bottom_dofs)
    bc_vals = np.zeros_like(bc_dofs)

    # ### Assemble and apply BCs globally
    A = np.zeros((N, N))
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        A[np.ix_(dofs, dofs)] += elementmatrix
    b = np.zeros((N,))
    A_full = A.copy()
    apply_bcs(A, b, bc_dofs, bc_vals)

    # ### Do the same via COO matrix
    data = []
    rows = []
    cols = []
    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)
        for l, x in enumerate(dofs):
            for k, y in enumerate(dofs):
                rows.append(x)
                cols.append(y)
                data.append(elementmatrix[l, k])

    K = coo_array((data, (rows, cols)), shape=(N, N))
    B = K.toarray()
    assert np.allclose(A_full, B)

    apply_bcs(B, b, bc_dofs, bc_vals)
    assert np.allclose(A, B)

    # ### Use COO matrix, but apply BCs locally
    diagonals = []
    data = []
    rows = []
    cols = []
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
                            data.append(1.0)
                            diagonals.append(x)
                        
                else:
                    rows.append(x)
                    cols.append(y)
                    data.append(elementmatrix[l, k])

    K = coo_array((data, (rows, cols)), shape=(N, N))
    assert np.allclose(A, K.toarray())


def test_inhom():
    """TODO: inhomogeneous BCs"""


if __name__ == "__main__":
    test()
