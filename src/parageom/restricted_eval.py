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
    sub_space_mask: np.ndarray

    def evaluate(self, expression: fem.Expression, values: Optional[np.ndarray] = None) -> np.ndarray:
        domain = self.quadrature_space.mesh
        cells = self.affected_cells

        if values is None:
            values = expression.eval(domain, cells)
        else:
            expression.eval(domain, cells, values)
        mask = self.sub_space_mask
        return values[mask]


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
    mask = np.full((parent_cells.size, len(q_points) * space.dofmap.bs), False, dtype=bool)

    rows = []
    cols = local_dofs

    for parent in affected_cells:
        child = cell_map.index(parent)
        rows.append(child)

    mask[rows, cols] = True

    return QuadratureSpaceRestriction(space, subspace, parent_cells, mask)


@pytest.mark.parametrize("value_shape", [(), (2,), (4,)])
def test_square(value_shape):
    nx = ny = 10
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=value_shape, degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_triangles = domain.topology.index_map(domain.topology.dim).size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global

    all_dofs = np.arange(dim_qspace_global, dtype=np.int32)
    num_magic_points = 23
    interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))


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

    qsr = build_restriction(Q, interp_dofs)

    print(f"choose {interp_dofs.size} / {dim_qspace_global} magic points")
    print(f"magic points {interp_dofs}")
    print(f"number of affected cells: {qsr.affected_cells.size}/ {num_triangles}")

    # let expression allocate space, values=None
    result = qsr.evaluate(expr)
    test = np.allclose(result, reference)
    assert test

    # use subspace function
    g = fem.Function(qsr.sub_space)
    values = g.x.array.reshape(np.unique(qsr.affected_cells).size, -1)
    qsr.evaluate(expr, values=values)
    test = np.allclose(reference, values[qsr.sub_space_mask])


def test_interval():
    num_cells = 10
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    V = fem.functionspace(domain, ("Lagrange", 2, ()))
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(), degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_interval = domain.topology.index_map(domain.topology.dim).size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global

    all_dofs = np.arange(dim_qspace_global, dtype=np.int32)
    num_magic_points = 9
    interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))

    f = fem.Function(V)
    f.interpolate(lambda x: 0.5 * x[0] ** 2) # type: ignore

    ufl_expr = ufl.grad(f)
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
    expr = fem.Expression(ufl_expr, q_points)

    q = fem.Function(Q)
    q.interpolate(expr)
    reference = q.x.array[:].copy()[interp_dofs]

    qsr = build_restriction(Q, interp_dofs)

    print(f"choose {interp_dofs.size} / {dim_qspace_global} magic points")
    print(f"magic points {interp_dofs}")
    print(f"number of affected cells: {qsr.affected_cells.size}/ {num_interval}")

    # let expression allocate space, values=None
    result = qsr.evaluate(expr)
    test = np.allclose(result, reference)
    assert test

    # use subspace function
    g = fem.Function(qsr.sub_space)
    values = g.x.array.reshape(np.unique(qsr.affected_cells).size, -1)
    qsr.evaluate(expr, values=values)
    test = np.allclose(reference, values[qsr.sub_space_mask])


if __name__ == "__main__":
    test_interval()
    test_square((2,))
