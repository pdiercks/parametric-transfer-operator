"""boundary condition restriction"""

from typing import Union, NamedTuple, Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from mpi4py import MPI
import dolfinx as df

from multi.misc import x_dofs_vectorspace


class BCGeom(NamedTuple):
    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    locator: Callable
    V: df.fem.FunctionSpace


class BCTopo(NamedTuple):
    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    entities: npt.NDArray[np.int32]
    entity_dim: int
    V: df.fem.FunctionSpace
    sub: Optional[int] = None


def _create_dirichlet_bcs(bcs: list[Union[BCGeom, BCTopo]]) -> list[df.fem.DirichletBC]:
    """Creates list of `df.fem.DirichletBC`.

    Args:
        bcs: The BC specification.
    """
    r = list()
    for nt in bcs:
        space = None
        if isinstance(nt, BCGeom):
            space = nt.V
            dofs = df.fem.locate_dofs_geometrical(nt.V, nt.locator)
        elif isinstance(nt, BCTopo):
            space = nt.V.sub(nt.sub) if nt.sub is not None else nt.V
            dofs = df.fem.locate_dofs_topological(nt.V, nt.entity_dim, nt.entities)
        else:
            raise TypeError
        try:
            bc = df.fem.dirichletbc(nt.value, dofs, space)
        except TypeError:
            bc = df.fem.dirichletbc(nt.value, dofs)
        r.append(bc)
    return r


def test():
    from .matrix_based_operator import affected_cells, build_dof_map

    domain = df.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    value_shape = (2, )
    V = df.fem.functionspace(domain, ("P", 1, value_shape))

    def bottom(x):
        return np.isclose(x[1], 0.0)

    # g = df.fem.Constant(domain, (df.default_scalar_type(9.), ) * value_shape[0])
    g = df.fem.Constant(domain, df.default_scalar_type(9.))

    tdim = domain.topology.dim
    fdim = tdim - 1

    # mybc = BCGeom(g, bottom, V)
    bottom_facets = df.mesh.locate_entities_boundary(domain, fdim, bottom)
    bottom_verts = df.mesh.locate_entities_boundary(domain, 0, bottom)
    mybc = BCTopo(g, bottom_verts, fdim, V, sub=0)
    bcs_V = _create_dirichlet_bcs([mybc])

    magic_dofs = np.array([0, 1, 6, 13], dtype=np.int32)
    cells = affected_cells(V, magic_dofs)

    submesh, cell_map, vertex_map, _ = df.mesh.create_submesh(domain, tdim, cells)
    V_r_source = df.fem.functionspace(submesh, V.ufl_element())

    # the submesh contains all cells sharing magic dofs
    # it may be that those cells contain facets of some boundary although the selected
    # magic dofs are not associated with those facets
    # Therefore, the restricted BC may have dofs on the original boundary.
    # Thus, it does not suffice to check via the dof coordinates as below.
    # A better check would be to assemble matrix, apply bc, restrict to magic dofs,
    # compare final entries of interest.

    # facets = np.zeros(mesh.num_facets(), dtype=np.uint)
    # facets[bc.markers()] = 1
    # facets_r = facets[parent_facet_indices]
    # sub_domains = df.MeshFunction('size_t', submesh, mesh.topology().dim() - 1)
    # sub_domains.array()[:] = facets_r
    #
    # bc_r = df.DirichletBC(V_r_source, bc.value(), sub_domains, 1, bc.method())

    submesh.topology.create_connectivity(tdim, fdim)
    c2f = submesh.topology.connectivity(tdim, fdim)
    sub_facets = set()
    for ci in range(submesh.topology.index_map(tdim).size_local):
        facets = c2f.links(ci)
        for fac in facets:
            sub_facets.add(fac)
    sub_facets = np.array(list(sorted(sub_facets)), dtype=np.int32)

    facet_mesh, facet_map, _, _ = df.mesh.create_submesh(submesh, fdim, sub_facets)

    num_verts = V.mesh.topology.index_map(0).size_local
    vertices = np.zeros(num_verts, dtype=np.int32)
    vertices[mybc.entities] = 99
    verts_r = vertices[vertex_map]
    vert_tags = np.nonzero(verts_r)[0]

    # num_facets = V.mesh.topology.index_map(fdim).size_local
    # facets = np.zeros(num_facets, dtype=np.int32)
    # facets[mybc.entities] = 99
    # facets_r = facets[facet_map]
    # facet_tags = np.nonzero(facets_r)[0]

    # rbc = BCTopo(g, facet_tags, fdim, V_r_source)
    rbc = BCTopo(g, vert_tags, fdim, V_r_source.sub(0))
    # r_dofs = df.fem.locate_dofs_topological(V_r_source, fdim, facet_tags)
    # rbc = df.fem.dirichletbc(g, r_dofs, V_r_source)

    bcs_V_r_range = _create_dirichlet_bcs([rbc])

    xdofs = x_dofs_vectorspace(V)
    other = x_dofs_vectorspace(V_r_source)

    dofs = bcs_V[0]._cpp_object.dof_indices()[0]
    rdofs = bcs_V_r_range[0]._cpp_object.dof_indices()[0]

    # check which magic_dofs are also bc dofs
    restricted_range_dofs = build_dof_map(V, cell_map, V_r_source, magic_dofs)
    is_member = np.in1d(magic_dofs, dofs)
    is_rmem = np.in1d(restricted_range_dofs, rdofs)

    err = xdofs[magic_dofs[is_member]] - other[restricted_range_dofs[is_rmem]]
    assert np.sum(err) < 1e-9

    # testing
    # build bc objects for V
    # build bc objects for Vsub
    # extract dof indices from bc objects
    # compare coordinates of bc dofs via V.tabulate_dof_coordinates()


if __name__ == "__main__":
    test()
