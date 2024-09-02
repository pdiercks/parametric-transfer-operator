"""beam check"""

import numpy as np
from mpi4py import MPI
import dolfinx as df
from multi.preprocessing import create_meshtags
from multi.boundary import plane_at, within_range
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.domain import RectangularDomain

from .tasks import example
from .matrix_based_operator import BCTopo, _create_dirichlet_bcs
from .stress_analysis import principal_stress_2d

E = 30e3
NU = 0.2
L = 1000.0
H = 100.0
nx = 1000
ny = 100

domain = df.mesh.create_rectangle(
    MPI.COMM_WORLD, np.array([[0.0, 0.0], [L, H]]), (nx, ny), cell_type=df.mesh.CellType.quadrilateral
)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1

V = df.fem.functionspace(domain, ("P", 2, (gdim,)))

top_marker = int(194)
top_locator = plane_at(H, "y")
facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
omega = RectangularDomain(domain, facet_tags=facet_tags)

mat = LinearElasticMaterial(gdim=gdim, E=E, NU=NU, plane_stress=False)
problem = LinearElasticityProblem(omega, V, mat)

# ### Dirichlet bcs are defined globally
x = domain.geometry.x
a = 100.
xmin = np.amin(x, axis=0)
xmax = np.amax(x, axis=0)
left = within_range([xmin[0], xmin[1], xmin[2]], [a / 2, xmin[1], xmin[2]])
u_origin = (df.default_scalar_type(0.0), df.default_scalar_type(0.0))
dirichlet_left = {
    "value": u_origin,
    "boundary": left,
    "entity_dim": 1,
    "sub": None,
}

right = within_range(
    [xmax[0] - a / 2, xmin[1], xmin[2]], [xmax[0], xmin[1], xmin[2]]
)
u_bottom_right = df.default_scalar_type(0.0)
dirichlet_right = {
    "value": u_bottom_right,
    "boundary": right,
    "entity_dim": 1,
    "sub": 1,
}

# Dirichlet BCs
entities_left = df.mesh.locate_entities_boundary(
    domain, fdim, dirichlet_left["boundary"]
)
bc_left = BCTopo(
    df.fem.Constant(V.mesh, dirichlet_left["value"]),
    entities_left,
    fdim,
    V,
    sub=dirichlet_left["sub"],
)
entities_right = df.mesh.locate_entities_boundary(
    domain, fdim, dirichlet_right["boundary"]
)
bc_right = BCTopo(
    df.fem.Constant(V.mesh, dirichlet_right["value"]),
    entities_right,
    fdim,
    V,
    sub=dirichlet_right["sub"],
)
bcs = _create_dirichlet_bcs((bc_left, bc_right))
for bc in bcs:
    problem.add_dirichlet_bc(bc)

# Neumann BCs
TY = -example.traction_y
traction = df.fem.Constant(
    domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
)
problem.add_neumann_bc(top_marker, traction)

problem.setup_solver()
u = problem.solve()

# output space
W = df.fem.functionspace(domain, ("P", 1, (gdim,)))
w = df.fem.Function(W, name="u")
w.interpolate(u)

# with df.io.XDMFFile(domain.comm, "beam.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(w)

principal_stress_2d(u, 2, mat)

breakpoint()
