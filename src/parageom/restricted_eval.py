from mpi4py import MPI
from dolfinx import fem, mesh, geometry
import basix
import ufl
import numpy as np


def find_cells(domain, points):
    # if points are not shared between cells, there should be no collisions
    tree = geometry.bb_tree(domain, domain.topology.dim)
    if not points.shape[1] == 3:
        points = points.T
    cells = geometry.compute_collisions_points(tree, points)
    return cells.array


def test():
    num_cells = 10
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    V = fem.functionspace(domain, ("Lagrange", 2, ()))
    q_deg = 2
    qe = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(), degree=q_deg)
    Q = fem.functionspace(domain, qe)

    num_qpoints = Q.dofmap.index_map.size_local
    dim_qspace = Q.dofmap.bs * Q.dofmap.index_map.size_local
    dim_qspace_global = Q.dofmap.bs * Q.dofmap.index_map.size_global
    print(f"{num_qpoints=}")
    print(f"{dim_qspace=}")
    print(f"{dim_qspace_global=}")

    f = fem.Function(V)
    f.interpolate(lambda x: 0.5 * x[0] ** 2) # type: ignore

    ufl_expr = ufl.grad(f)
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
    expr = fem.Expression(ufl_expr, q_points)

    q = fem.Function(Q)
    q.interpolate(expr)

    # expr happens to be the same as integration points in physical space
    reference = q.x.array[:].copy()

    # q.interpolate(lambda x: x[0])
    # phy_x = q.x.array[:].copy()

    interp_dofs = np.array([1, 4, 7, 10, 11, 17], dtype=np.int32)
    print(reference[interp_dofs])

    # fill with zeroes such that points.shape equals (num_points, 3)
    # points = np.pad(reference[interp_dofs][:, np.newaxis], [(0, 0), (0, 2)], mode='constant')
    xdofs = Q.tabulate_dof_coordinates()
    points = xdofs[interp_dofs]
    cells = find_cells(domain, points)

    # cells may not be unique
    cells = np.unique(cells)

    values = expr.eval(domain, cells)

    # q.x.array[:] = 0.
    # q.interpolate(lambda x: x[0], cells=cells)
    # other = q.x.array[:]

    # the difference between fem.Expression.eval(domain, cells) & function.interpolate(expr, cells)
    # is the size of the return value; or rather that function.interpolate has no return value
    breakpoint()


    # values.shape = (num_cells, num_qpoints_per_cell)
    # need to pick for each cell, the value according to interp_dofs
    # need map from global interp_dofs to local dof value

    # qmap = Q.dofmap
    # qmap.cell_dofs maps cell index to global dofs (non-blocked); would need to consider qmap.bs for vector valued spaces
    # qmap.list returns array of shape (num_cells, num_qpoints_per_cell)
    # thus np.where(qmap.list == interp_dofs[0]) would return (cell, local_index) = (0, 1)
    # now vectorize?

    qmap = Q.dofmap
    # bs - is e.g. the number of vector components of the field stored at each point
    num_repeats = Q.dofmap.bs * len(q_points)
    mask = qmap.list[cells].flatten() == np.repeat(interp_dofs, num_repeats)
    restricted_eval = values[mask.reshape(values.shape)]
    if np.allclose(restricted_eval, reference[interp_dofs]):
        print("Success!")


if __name__ == "__main__":
    test()
