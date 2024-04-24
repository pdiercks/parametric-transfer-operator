import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import fem

from multi.debug import plot_modes
from multi.io import read_mesh
from multi.domain import RectangularSubdomain
from multi.problems import LinElaSubProblem
from multi.materials import LinearElasticMaterial
from .tasks import example

domain, _, _ = read_mesh(example.parent_unit_cell, MPI.COMM_SELF, gdim=2)

omega = RectangularSubdomain(11, domain)
omega.create_coarse_grid(1)
omega.create_boundary_grids()

material = LinearElasticMaterial(gdim=2, E=1., NU=0.3)
V = fem.functionspace(domain, ("P", 2, (2,)))
p = LinElaSubProblem(omega, V, material)

p.setup_edge_spaces()

edge = "bottom"
edge_space = p.edge_spaces["fine"][edge]

data = np.load(example.fine_scale_edge_modes_npz(0, "hapod", "normal", "left"))
modes = data[edge]
num_modes = 4
component = "x"

plot_modes(edge_space, edge, modes, component, np.s_[:num_modes], show=False)
plt.savefig("./modes.png")
