import numpy as np

from mpi4py import MPI
from dolfinx import fem, mesh
import basix
import ufl

num_cells = 3
domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)

q_deg = 1
qele = basix.ufl.quadrature_element(domain.topology.cell_name(), value_shape=(), degree=q_deg)
Q = fem.functionspace(domain, qele)


subcells = np.array([1], dtype=np.int32)
submesh, cell_map, _, _ = mesh.create_submesh(domain, domain.topology.dim, subcells)
Qsub = fem.functionspace(submesh, qele)

V = fem.functionspace(domain, ("P", 2, ()))
f = fem.Function(V)
f.interpolate(lambda x: 0.5 * x[0] ** 2)

ufle = ufl.grad(f)
basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
q_points, _ = basix.make_quadrature(basix_celltype, q_deg)
expression = fem.Expression(ufle, q_points)

values = expression.eval(domain, subcells)
other = expression.eval(submesh, np.array([0], dtype=np.int32))
print(f"{values=}") # array([[0.5]])
print(f"{other=}") # array([[0.16666667]])
