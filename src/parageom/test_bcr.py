"""boundary condition restriction"""

import numpy as np

from mpi4py import MPI
import dolfinx as df
from .matrix_based_operator import (
    _affected_cells,
    _build_dof_map,
    BCGeom,
    BCTopo,
    _create_dirichlet_bcs,
    _restrict_bc_topo,
    SubmeshWrapper
)
from multi.boundary import point_at


def test(bc_specs, V):
    bcs = _create_dirichlet_bcs(bc_specs)
    ndofs = V.dofmap.bs * V.dofmap.index_map.size_local

    # restricted code to get data structures required
    magic_dofs = set()
    nmagic = 13
    while len(magic_dofs) < nmagic:
        magic_dofs.add(np.random.randint(0, ndofs))
    magic_dofs = np.array(list(sorted(magic_dofs)), dtype=np.int32)
    # magic_dofs = np.array([1, 7, 15, 20, 23, 44], dtype=np.int32)
    cells = _affected_cells(V, magic_dofs)

    source_dofmap = V.dofmap
    source_dofs = set()
    for cell_index in cells:
        local_dofs = source_dofmap.cell_dofs(cell_index)
        for ld in local_dofs:
            for b in range(source_dofmap.bs):
                source_dofs.add(ld * source_dofmap.bs + b)
    source_dofs = np.array(sorted(source_dofs), dtype=magic_dofs.dtype)

    tdim = domain.topology.dim
    submesh_wrapper = SubmeshWrapper(*df.mesh.create_submesh(
        domain, tdim, cells
    ))
    fdim = tdim - 1
    submesh = submesh_wrapper.mesh
    submesh.topology.create_connectivity(tdim, fdim)
    # make sure entity-to-cell connectivities are computed
    submesh.topology.create_connectivity(fdim, tdim)
    submesh.topology.create_connectivity(0, tdim)

    V_r_source = df.fem.functionspace(submesh, V.ufl_element())
    V_r_range = df.fem.functionspace(submesh, V.ufl_element())
    interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
        V_r_source.mesh, V_r_source.element, V.mesh
    )

    r_bcs = list()
    for nt in bc_specs:
        if isinstance(nt.value, df.fem.Function):
            # interpolate function to submesh
            g = df.fem.Function(V_r_source, name=nt.value.name)
            g.interpolate(nt.value, nmm_interpolation_data=interp_data)
        else:
            # g is of type df.fem.Constant or np.ndarray
            g = nt.value

        if isinstance(nt, BCGeom):
            rbc = BCGeom(g, nt.locator, V_r_source)
        elif isinstance(nt, BCTopo):
            rbc = _restrict_bc_topo(V.mesh, submesh_wrapper, nt, g, V_r_source)
        else:
            raise TypeError
        r_bcs.append(rbc)


    bcs_V_r_range = _create_dirichlet_bcs(r_bcs)

    # check which magic_dofs are also bc dofs
    r_source_dofs = _build_dof_map(V, V_r_source, source_dofs, interp_data)
    r_range_dofs = _build_dof_map(V, V_r_range, magic_dofs, interp_data)

    assert r_range_dofs.size == magic_dofs.size

    u = df.fem.Function(V)
    u.x.array[:] = np.random.rand(ndofs)
    df.fem.set_bc(u.x.array, bcs)

    u_r = u.x.array[source_dofs[np.argsort(r_source_dofs)]].copy()
    df.fem.set_bc(u_r, bcs_V_r_range)

    def get_dof_indices(bcs):
        r = []
        for bc in bcs:
            ds = bc._cpp_object.dof_indices()[0]
            r.append(ds)
        return np.hstack(r)

    bcdofs = get_dof_indices(bcs)
    r_bcdofs = get_dof_indices(bcs_V_r_range)

    num_magic_in_bcs = 0
    for dof in magic_dofs:
        if dof in bcdofs:
            num_magic_in_bcs += 1

    num_magic_in_rbcs = 0
    for dof in r_range_dofs:
        if dof in r_bcdofs:
            num_magic_in_rbcs += 1

    if not num_magic_in_bcs == num_magic_in_rbcs:
        breakpoint()

    ref = u.x.array[magic_dofs]
    result = u_r[r_range_dofs]
    err = ref - result
    assert np.sum(np.abs(err)) < 1e-12


if __name__ == "__main__":
    nx = ny = 19
    domain = df.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    value_shape = (2,)
    degree = 2
    V = df.fem.functionspace(domain, ("P", degree, value_shape))

    def bottom(x):
        return np.isclose(x[1], 0.0)

    def right(x):
        return np.isclose(x[0], 1.0)

    def left(x):
        return np.isclose(x[0], 0.0)

    # geometrical bcs
    u_bottom: df.fem.Function = df.fem.Function(V) # type: ignore
    u_bottom.interpolate(lambda x: (np.sin(2 * np.pi * x[0]), x[1]))
    u_right = df.fem.Constant(domain, (df.default_scalar_type(12.7), ) * value_shape[0])
    bcs_geom = [
            BCGeom(u_bottom, bottom, V), BCGeom(u_right, right, V)
            ]
    test(bcs_geom, V)
    print("geom passed")

    # topological bcs
    tdim = domain.topology.dim
    fdim = tdim - 1
    bottom_facets = df.mesh.locate_entities_boundary(domain, fdim, bottom)
    left_facets = df.mesh.locate_entities_boundary(domain, fdim, left)
    upper_right = point_at([1., 1.])
    upper_right_vertex = df.mesh.locate_entities_boundary(domain, 0, upper_right)
    u_top_x = np.array((123.), dtype=df.default_scalar_type)
    bcs_topo = [
        BCTopo(u_right, bottom_facets, fdim, V),
        BCTopo(df.default_scalar_type(78.), left_facets, fdim, V, sub=1),
        BCTopo(u_top_x, upper_right_vertex, 0, V, sub=0)
    ]
    test(bcs_topo, V)
    print("topo passed")
