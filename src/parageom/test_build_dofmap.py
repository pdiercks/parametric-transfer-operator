from mpi4py import MPI
from dolfinx import fem, mesh
import numpy as np
from multi.misc import x_dofs_vectorspace


def build_dof_map(V, cell_map, V_r, dofs) -> np.ndarray:
    """Computes interpolation dofs of V_r.

    Args:
        V: The function space.
        cell_map: Indices of parent cells.
        V_r: The restricted space.
        dofs: Interpolation DOFs of V.
    """
    assert V.dofmap.bs == V_r.dofmap.bs

    subdomain = V_r.mesh
    tdim = subdomain.topology.dim
    num_cells = subdomain.topology.index_map(tdim).size_local

    children = set()

    for cell in range(num_cells):
        parent_cell = cell_map[cell]
        parent_dofs_local = V.dofmap.cell_dofs(parent_cell)
        child_dofs_local = V_r.dofmap.cell_dofs(cell)

        for b in range(V.dofmap.bs):
            for pdof, cdof in zip(parent_dofs_local, child_dofs_local):
                if pdof * V.dofmap.bs + b in dofs:
                    children.add(cdof * V.dofmap.bs + b)
    return np.array(list(children), dtype=dofs.dtype)


def test(domain, value_shape):

    # TODO
    # other domain
    # other value shape

    V = fem.functionspace(domain, ("P", 1, value_shape))

    ndofs = V.dofmap.bs * V.dofmap.index_map.size_local
    magic_dofs = set()
    nmagic = 12
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
    
    source_dofmap = V.dofmap
    source_dofs = set()
    for cell_index in affected_cell_indices:
        ldofs = source_dofmap.cell_dofs(cell_index)
        for ld in ldofs:
            for b in range(source_dofmap.bs):
                source_dofs.add(ld * source_dofmap.bs + b)

    source_dofs = np.array(sorted(source_dofs), dtype=np.int32)

    tdim = domain.topology.dim
    submesh, cell_map, _, _ = mesh.create_submesh(domain, tdim, affected_cell_indices)
    Vsub = fem.functionspace(submesh, V.ufl_element())

    r_source_dofs = build_dof_map(V, cell_map, Vsub, source_dofs)
    assert source_dofs.size == r_source_dofs.size

    # compare dof coordinates
    xdofs = x_dofs_vectorspace(V)[source_dofs[np.argsort(r_source_dofs)]]
    other = x_dofs_vectorspace(Vsub)


    diff = np.abs(xdofs - other)
    if domain.topology.dim == 2:
        breakpoint()
        # FIXME
        # this does not work in 2D
    assert np.sum(diff) < 1e-9



if __name__ == "__main__":
    num_intervals = 20
    unit_interval = mesh.create_unit_interval(MPI.COMM_WORLD, num_intervals)

    test(unit_interval, ())
    test(unit_interval, (2,))

    num_triangles = 8
    unit_square = mesh.create_unit_square(MPI.COMM_WORLD, num_triangles, num_triangles)

    test(unit_square, ())
