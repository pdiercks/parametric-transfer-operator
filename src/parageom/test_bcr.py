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

    # ### Determine Dirichlet entities on submesh
    # FIXME: need to do this for entity_dim in BCTopo, not just fdim


    # only use the entities in the BCTopo spec to create a submesh !
    # Would need to do this for every BC though,
    # but this should not pose a problem
    # c2f = submesh.topology.connectivity(tdim, fdim)
    # sub_facets = set()
    # for ci in range(submesh.topology.index_map(tdim).size_local):
    #     facets = c2f.links(ci)
    #     for fac in facets:
    #         # TODO
    #         # somehow determine if on_boundary? here?
    #         breakpoint()
    #         print("on boundary?")
    #         sub_facets.add(fac)
    # sub_facets = np.array(list(sorted(sub_facets)), dtype=np.int32)
    # _, parent_facet_indices, _, _ = df.mesh.create_submesh(
    #     submesh, fdim, sub_facets
    # )

    # determine parent facets on submesh boundary
    # def everywhere(x):
    #     return np.full(x[0].shape, True, dtype=bool)

    # FIXME this is the whole boundary
    # but only 'bottom' is needed according to nt.entities
    # boundary_facets = df.mesh.locate_entities_boundary(submesh, fdim, everywhere)
    # boundary_vertices = df.mesh.locate_entities_boundary(submesh, 0, everywhere)

    # I have all parent facet indices
    # nt.entities has all parent facets on the Dirichlet boundary
    # How can I determine which of nt.entities are also on the boundary of submesh?

    # parent_boundary_entities = {fdim: np.intersect1d(parent_facet_indices, boundary_facets), 0: np.intersect1d(parent_vertex_indices, boundary_vertices)}
    # parent_boundary_entities = {fdim: parent_facet_indices, 0: parent_vertex_indices}

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
    u.x.array[:] = np.zeros(ndofs)
    # u.x.array[:] = np.random.rand(ndofs)
    df.fem.set_bc(u.x.array, bcs)
    # print(np.sum(u.x.array))

    u_r = np.zeros(source_dofs.size)
    df.fem.set_bc(u_r, bcs_V_r_range)
    # print(np.sum(u_r)) # this should be smaller than value above
    # somehow I get to many bc dofs in the restriction
    # number of bc dofs can only decrease


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
