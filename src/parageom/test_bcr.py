"""boundary condition restriction"""

import numpy as np
from mpi4py import MPI
import dolfinx as df


def test():
    from .matrix_based_operator import affected_cells, blocked, build_dof_map, unblock

    domain = df.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    value_shape = (2, )
    V = df.fem.functionspace(domain, ("P", 1, value_shape))
    # V = df.fem.functionspace(domain, ("P", 1))
    S = V
    R = V

    def bottom(x):
        return np.isclose(x[1], 0.0)

    # def right(x):
    #     return np.isclose(x[0], 1.0)

    bc_dofs_bottom = df.fem.locate_dofs_geometrical(V, bottom)
    # bc_dofs_right = df.fem.locate_dofs_geometrical(V, right)
    # g = df.fem.Function(V)
    # g.interpolate(lambda x: x[0])
    g = df.fem.Constant(domain, (df.default_scalar_type(9.), ) * value_shape[0])
    bc = df.fem.dirichletbc(g, bc_dofs_bottom, V)
    # bc = df.fem.dirichletbc(g, bc_dofs_right)

    ddofs, num_ddofs = bc._cpp_object.dof_indices()

    magic_dofs = np.array([0, 1, 6, 13], dtype=np.int32)
    cells = affected_cells(V, magic_dofs)

    # magic dofs contains (blocked) dofs
    # however, for some vertex (dof) it may only contain dof_y instead of (dof_x, dof_y) (bs=2)
    # therefore, unblocking does not work correctly?

    source_dofmap = S.dofmap
    source_dofs = set()
    for cell_index in cells:
        local_dofs = blocked(source_dofmap.cell_dofs(cell_index), source_dofmap.bs)
        source_dofs.update(local_dofs)
    source_dofs = np.array(sorted(source_dofs), dtype=local_dofs.dtype)

    tdim = domain.topology.dim
    # fdim = tdim - 1
    submesh, cell_map, _, _ = df.mesh.create_submesh(domain, tdim, cells)
    V_r_source = df.fem.functionspace(submesh, S.ufl_element())
    V_r_range = df.fem.functionspace(submesh, R.ufl_element())

    restricted_source_dofs = build_dof_map(S, cell_map, V_r_source, source_dofs)
    restricted_range_dofs = build_dof_map(R, cell_map, V_r_range, magic_dofs)

    mask = np.nonzero(source_dofs[:, np.newaxis] - magic_dofs[np.newaxis, :] == 0)[0]
    # restricted_dirichlet_dofs = restricted_source_dofs[mask]
    rdd = restricted_source_dofs[mask]

    is_member = np.in1d(magic_dofs, ddofs)
    if np.any(is_member):
        # map magic_dofs to reduced space
        # unblock to be able to define dirichletbc in reduced space
        print("magic")
        # TODO if dofmap.bs > 1 we need to "unblock" rdd
        # TODO depending on type of value, need to interpolate or something

        # this approach will also never work for component-wise dirichlet bcs

        value = bc.g
        bs = V_r_source.dofmap.bs
        un_rdd = unblock(rdd, bs) # <-- Need to know how many pairs of dofs
        bcr = df.fem.dirichletbc(value, un_rdd, V_r_source) # value=Constant,np.ndarray
        breakpoint()
        # bcr = df.fem.dirichletbc(value, rdd) # value=Function
    else:
        print("nothing to do")
        # do nothing

    # OTHER APPROACH
    # from affected cells determine facets
    # and add them to affected facets if they also are part of the boundary
    # However, this approach does not work, because we may get boundary facets
    # that are not connected to the magic dofs.

    # KISS
    # support geometrical BCs for now
    # let the user provide bcs as namedtuple("BC", ["value", "boundary_marker"])


if __name__ == "__main__":
    test()
