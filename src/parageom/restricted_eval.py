from typing import Optional
from dataclasses import dataclass

from mpi4py import MPI
from dolfinx import fem, mesh
import basix
import ufl
import numpy as np
import pytest


@dataclass
class QuadratureSpaceRestriction:
    quadrature_space: fem.FunctionSpace
    sub_space: fem.FunctionSpace
    affected_cells: np.ndarray
    local_dofs: np.ndarray
    sub_space_mask: np.ndarray

    def evaluate(self, expression: fem.Expression, values: Optional[np.ndarray] = None):
        domain = self.quadrature_space.mesh
        cells = self.affected_cells
        ldofs = self.local_dofs

        # FIXME
        # evaluate fem.Expression over submesh instead of full mesh
        # mask is not correctly build
        # domain = self.sub_space.mesh
        # num_cells = domain.topology.index_map(domain.topology.dim).size_local
        # cells = np.arange(num_cells, dtype=np.int32)

        if values is None:
            values = expression.eval(domain, cells)
        else:
            # FIXME
            # affected cells may contain duplicate cells
            # therefore something like
            # g = fem.Function(self.sub_space)
            # values = g.x.array.reshape(...)
            # will not work ...
            expression.eval(domain, cells, values)

        # FIXME
        # mask = self.sub_space_mask
        # return values[mask]
        return values[np.arange(cells.size), ldofs]


# ### Note
# I think this is a bug with submesh
# expression.eval(submesh, all_cells) returns values < submesh.geometry.x.min()


def build_restriction(space, magic_points) -> QuadratureSpaceRestriction:
    """Determines affected cells and local dof indices.

    Args:
        space: The quadrature space.
        magic_points: Interpolation (quadrature) points selected by the greedy
        algorithm.

    """

    affected_cells = []
    local_dofs = []

    qmap = space.dofmap
    domain = space.mesh
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    for cell in range(num_cells):
        cell_dofs = qmap.cell_dofs(cell)
        for locald, globald in enumerate(cell_dofs):
            for b in range(qmap.bs):
                if globald * qmap.bs + b in magic_points:
                    affected_cells.append(cell)
                    local_dofs.append(locald * qmap.bs + b)
    affected_cells = np.array(affected_cells, dtype=np.int32)
    local_dofs = np.array(local_dofs, dtype=np.int32)

    parent_cells = np.unique(affected_cells)

    submesh, cell_map, _, _ = mesh.create_submesh(domain, domain.topology.dim, parent_cells)
    # cell_map: maps child to parent cell
    subspace = fem.functionspace(submesh, space.ufl_element())

    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, space.ufl_element().degree)
    mask = np.full((parent_cells.size, len(q_points)), False, dtype=bool)

    rows = []
    cols = local_dofs

    for parent in affected_cells:
        child = cell_map.index(parent)
        rows.append(child)

    mask[rows, cols] = True
    breakpoint()

    return QuadratureSpaceRestriction(space, subspace, affected_cells, local_dofs, mask)


@pytest.mark.parametrize("value_shape", [(), (2,), (4,)])
def test_square(value_shape):
    nx = ny = 10
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=value_shape, degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_triangles = domain.topology.index_map(domain.topology.dim).size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global
    print(f"number of triangle cells: {num_triangles}")

    all_dofs = np.arange(dim_qspace_global, dtype=np.int32)
    num_magic_points = 23
    interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))
    print(f"choose {interp_dofs.size} / {dim_qspace_global} magic points")
    print(interp_dofs)


    ufl_expr = None

    if value_shape == ():
        V = fem.functionspace(domain, ("Lagrange", 2, ()))
        f = fem.Function(V)
        f.interpolate(lambda x: x[0] * x[1]) # type: ignore
        ufl_expr = ufl.grad(f)[0]
    elif value_shape == (2,):
        V = fem.functionspace(domain, ("Lagrange", 2, ()))
        f = fem.Function(V)
        f.interpolate(lambda x: x[0] * x[1]) # type: ignore
        ufl_expr = ufl.grad(f)
    elif value_shape == (4,):
        V = fem.functionspace(domain, ("Lagrange", 2, (2,)))
        f = fem.Function(V)
        f.interpolate(lambda x: (x[0], x[1])) # type: ignore
        e = ufl.sym(ufl.grad(f))
        ufl_expr = ufl.as_vector([
            e[0, 0], e[1, 1], 0.0, e[0, 1]
            ])
    else:
        raise NotImplementedError

    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
    expr = fem.Expression(ufl_expr, q_points)

    q = fem.Function(Q)
    q.interpolate(expr)
    reference = q.x.array[:].copy()[interp_dofs]
    # TODO test other cases
    # e.g. where magic points in the same cell are selected ...


    restr = build_restriction(Q, interp_dofs)
    breakpoint()
    # cells, local_magic = affected_cells(Q, interp_dofs)

    # NOTE
    # the ordering of cells and local_magic does not follow ordering
    # of interp_dofs.
    # I think it should be safe to simply sort interp_dofs.

    print(f"number of affected cells: {np.unique(cells).size} / {num_triangles}")
    print(f"{cells=}")
    print(f"{local_magic=}")
    assert cells.shape == local_magic.shape
    values = expr.eval(domain, cells)
    # should I build a subspace of Q to be able to instantiate reduced function that is used to allocate space for output?
    restr_eval = values[np.arange(cells.size), local_magic]

    test = np.allclose(restr_eval, reference)
    assert test


def test_interval():
    num_cells = 10
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    V = fem.functionspace(domain, ("Lagrange", 2, ()))
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(), degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_interval = domain.topology.index_map(domain.topology.dim).size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global
    print(f"number of interval cells: {num_interval}")

    all_dofs = np.arange(dim_qspace_global, dtype=np.int32)
    num_magic_points = 3
    # interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))
    interp_dofs = np.array([4, 5, 8], dtype=np.int32)
    print(f"choose {interp_dofs.size} / {dim_qspace_global} magic points")
    print(interp_dofs)

    f = fem.Function(V)
    f.interpolate(lambda x: 0.5 * x[0] ** 2) # type: ignore

    ufl_expr = ufl.grad(f)
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
    expr = fem.Expression(ufl_expr, q_points)

    q = fem.Function(Q)
    q.interpolate(expr)
    reference = q.x.array[:].copy()[interp_dofs]
    # TODO test other cases
    # e.g. where magic points in the same cell are selected ...

    qsr = build_restriction(Q, interp_dofs)
    g = fem.Function(qsr.sub_space)
    result = qsr.evaluate(expr)
    breakpoint()

    # NOTE
    # the ordering of cells and local_magic does not follow ordering
    # of interp_dofs.
    # I think it should be safe to simply sort interp_dofs.

    cells = qsr.affected_cells
    local_magic = qsr.local_dofs
    print(f"number of affected cells: {np.unique(cells).size} / {num_interval}")
    print(f"{cells=}")
    print(f"{local_magic=}")
    assert cells.shape == local_magic.shape

    test = np.allclose(result, reference)
    assert test


if __name__ == "__main__":
    test_interval()
    # test_square()
