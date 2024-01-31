from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from dolfinx import fem
from basix.ufl import element
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
import numpy as np


def main():
    from ..tasks import beam

    unit_cell_msh = beam.unit_cell_grid.as_posix()
    gdim = beam.gdim
    domain, _, _ = gmshio.read_from_msh(unit_cell_msh, MPI.COMM_SELF, gdim=gdim)

    fe = element("P", domain.basix_cell(), beam.fe_deg, shape=(gdim,))
    V = fem.functionspace(domain, fe)
    source = FenicsxVectorSpace(V)
    viz = FenicsxVisualizer(source)

    # need displacment for warp by vector
    u = fem.Function(V)
    def expression(x):
        return (np.sin(np.pi * x[0]) * np.sinh(np.pi * x[1]), np.cos(np.pi*x[0]))
    u.interpolate(expression)

    # distr = "normal"
    # config = "right"
    # data = np.load(beam.loc_pod_modes(distr, config))
    # U = source.from_numpy(data)
    U = source.make_array([u.vector])
    viz.visualize(U, filename="./test.bp")


if __name__ == "__main__":
    main()
