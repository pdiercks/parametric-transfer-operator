import numpy as np


def blocked(dofs: np.ndarray, bs: int):
    ndofs = dofs.size
    blocked = np.zeros((ndofs, bs), dtype=dofs.dtype)
    for i in range(bs):
        blocked[:, i] = i
    r = blocked + np.repeat(dofs[:, np.newaxis], bs, axis=1) * bs
    return r.flatten()

# Note
# We have to loop over the full mesh to determine affected cells.
# Then we can build the submesh and subspace.


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

    parents = []
    children = []

    for cell in range(num_cells):
        parent_cell = cell_map[cell]
        parent_dofs = blocked(V.dofmap.cell_dofs(parent_cell), V.dofmap.bs)
        child_dofs = blocked(V_r.dofmap.cell_dofs(cell), V_r.dofmap.bs)

        parents.append(parent_dofs)
        children.append(child_dofs)
    parents = np.unique(np.hstack(parents))
    children = np.unique(np.hstack(children))
    indx = np.nonzero(parents[:, np.newaxis] - dofs[np.newaxis, :]==0)[0]
    return children[indx]


if __name__ == "__main__":
    from mpi4py import MPI
    from dolfinx import fem, mesh, default_scalar_type
    import ufl
    domain = mesh.create_unit_interval(MPI.COMM_SELF, 5)
    V = fem.functionspace(domain, ("P", 1, (2,)))
    magic_dofs = np.array([4, 5, 10], dtype=np.int32)
    
    range_dofmap = V.dofmap
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    affected_cell_indices = set()
    for cell_index in range(num_cells):
        local_dofs = range_dofmap.cell_dofs(cell_index)
        # dofmap.cell_dofs() returns non-blocked dof indices
        for ld in blocked(local_dofs, range_dofmap.bs):
            if ld in magic_dofs:
                affected_cell_indices.add(cell_index)
                continue
    affected_cell_indices = np.array(list(sorted(affected_cell_indices)), dtype=np.int32)
    
    tdim = domain.topology.dim
    submesh, cell_map, _, _ = mesh.create_submesh(domain, tdim, affected_cell_indices)
    Vsub = fem.functionspace(submesh, V.ufl_element())
    map = build_dof_map(V, cell_map, Vsub, magic_dofs)

    v = ufl.TestFunction(V)
    c = fem.Constant(domain, (default_scalar_type(2.), default_scalar_type(-4.)))
    form = ufl.inner(v, c) * ufl.dx
    vector = fem.assemble_vector(fem.form(form))

    # restriced evaluation
    v = ufl.TestFunction(Vsub)
    c = fem.Constant(Vsub.mesh, (default_scalar_type(2.), default_scalar_type(-4.)))
    form = ufl.inner(v, c) * ufl.dx
    other = fem.assemble_vector(fem.form(form))

    assert np.allclose(vector.array[magic_dofs], other.array[map])
