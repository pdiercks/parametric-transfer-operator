from mpi4py import MPI
from dolfinx import fem, mesh
import basix
import ufl
import numpy as np


def affected_cells(qspace, magic_points):
    # TODO add docstring

    # this function should be called only once
    # in the offline phase

    # return value should be stored (dataclass?)
    # for restricted evaluation in the online phase

    affected_cells = []
    local_dofs = []

    qmap = qspace.dofmap
    domain = qspace.mesh
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    for cell in range(num_cells):
        cell_dofs = qmap.cell_dofs(cell)
        for locald, globald in enumerate(cell_dofs):
            for b in range(qmap.bs):
                if globald * qmap.bs + b in magic_points:
                    affected_cells.append(cell)
                    local_dofs.append(locald * qmap.bs + b)
    return np.array(affected_cells, dtype=np.int32), np.array(
            local_dofs, dtype=np.int32)


def test_square():
    nx = ny = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    V = fem.functionspace(domain, ("Lagrange", 2, ()))
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(2,), degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_triangles = domain.topology.index_map(domain.topology.dim).size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global
    print(f"number of triangle cells: {num_triangles}")

    all_dofs = np.arange(dim_qspace_global, dtype=np.int32)
    num_magic_points = 23
    interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))
    print(f"choose {interp_dofs.size} / {dim_qspace_global} magic points")
    print(interp_dofs)

    f = fem.Function(V)
    f.interpolate(lambda x: x[0] * x[1]) # type: ignore

    ufl_expr = ufl.grad(f)
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
    expr = fem.Expression(ufl_expr, q_points)

    q = fem.Function(Q)
    q.interpolate(expr)
    reference = q.x.array[:].copy()[interp_dofs]
    # TODO test other cases
    # e.g. where magic points in the same cell are selected ...


    cells, local_magic = affected_cells(Q, interp_dofs)

    # NOTE
    # the ordering of cells and local_magic does not follow ordering
    # of interp_dofs.
    # I think it should be safe to simply sort interp_dofs.

    print(f"number of affected cells: {np.unique(cells).size} / {num_triangles}")
    print(f"{cells=}")
    print(f"{local_magic=}")
    assert cells.shape == local_magic.shape
    values = expr.eval(domain, cells)
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
    num_magic_points = 13
    interp_dofs = np.sort(np.unique(np.random.choice(all_dofs, size=num_magic_points)))
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

    cells, local_magic = affected_cells(Q, interp_dofs)

    # NOTE
    # the ordering of cells and local_magic does not follow ordering
    # of interp_dofs.
    # I think it should be safe to simply sort interp_dofs.

    print(f"number of affected cells: {np.unique(cells).size} / {num_interval}")
    print(f"{cells=}")
    print(f"{local_magic=}")
    assert cells.shape == local_magic.shape
    values = expr.eval(domain, cells)
    restr_eval = values[np.arange(cells.size), local_magic]

    test = np.allclose(restr_eval, reference)
    assert test


if __name__ == "__main__":
    test_interval()
    test_square()
