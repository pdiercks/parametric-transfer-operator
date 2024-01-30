from pathlib import Path
import numpy as np
from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem
from dolfinx.io import gmshio

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from .tasks import beam

gdim = 2
domain, ct, ft = gmshio.read_from_msh(beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim)
fe = element("P", domain.basix_cell(), beam.fe_deg, shape=(gdim,))
V = fem.functionspace(domain, fe)

root = Path("/home/pdiercks/projects/2023_04_opt_am_concrete/muto")
stuffy = root / "stuff.npy"
data = np.load(stuffy.as_posix())

source = FenicsxVectorSpace(V)
U = source.from_numpy(data)

viz = FenicsxVisualizer(source)
viz.visualize(U, filename=stuffy.with_suffix(".xdmf"))
