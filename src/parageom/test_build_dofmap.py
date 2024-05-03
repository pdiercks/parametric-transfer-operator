from mpi4py import MPI
from dolfinx import fem, mesh
import numpy as np
from multi.misc import x_dofs_vectorspace


def test(domain, value_shape):
    from .matrix_based_operator import _build_dof_map
    V = fem.functionspace(domain, ("P", 1, value_shape))

    ndofs = V.dofmap.bs * V.dofmap.index_map.size_local
    magic_dofs = set()
    nmagic = 2
    while len(magic_dofs) < nmagic:
        magic_dofs.add(np.random.randint(0, ndofs))

    magic_dofs = np.array(list(magic_dofs), dtype=np.int32)
    
    range_dofmap = V.dofmap
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    affected_cell_indices = set()
    for cell_index in range(num_cells):
        local_dofs = range_dofmap.cell_dofs(cell_index)
        for ld in local_dofs:
            for b in range(range_dofmap.bs):
                if ld * range_dofmap.bs + b in magic_dofs:
                    affected_cell_indices.add(cell_index)
                    continue
    affected_cell_indices = np.array(list(sorted(affected_cell_indices)), dtype=np.int32)
    

    tdim = domain.topology.dim
    submesh, parent_cells, _, _ = mesh.create_submesh(domain, tdim, affected_cell_indices)
    Vsub = fem.functionspace(submesh, V.ufl_element())

    # ### DOFs
    assert Vsub.dofmap.bs == V.dofmap.bs
    block_size = V.dofmap.bs

    source_dofs = set()
    # what about ghost cells?
    # the below assumes source=range space
    num_affected_cells = submesh.topology.index_map(tdim).size_local
    for cell_index in range(num_affected_cells):
        parent_dofs = V.dofmap.cell_dofs(parent_cells[cell_index])
        for pdof in parent_dofs:
            for b in range(block_size):
                source_dofs.add(pdof * block_size + b)

    source_dofs = np.array(list(sorted(source_dofs)), dtype=np.int32)

    interp_data = fem.create_nonmatching_meshes_interpolation_data(
                Vsub.mesh,
                Vsub.element,
                V.mesh)

    r_source_dofs = _build_dof_map(V, Vsub, source_dofs, interp_data)

    # compare dof coordinates
    xdofs = x_dofs_vectorspace(V)[source_dofs[np.argsort(r_source_dofs)]]
    other = x_dofs_vectorspace(Vsub)


    diff = np.abs(xdofs - other)
    assert np.sum(diff) < 1e-9



if __name__ == "__main__":
    num_intervals = 20
    unit_interval = mesh.create_unit_interval(MPI.COMM_WORLD, num_intervals)

    test(unit_interval, ())
    test(unit_interval, (2,))

    num_triangles = 8
    unit_square = mesh.create_unit_square(MPI.COMM_WORLD, num_triangles, num_triangles)

    test(unit_square, ())
    test(unit_square, (2,))
    test(unit_square, (3,))
