import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from definitions import Example


def main(resolution, degree):
    ex = Example(name="beam", resolution=resolution)
    domain = preprocessing(ex)

    # gdim = domain.ufl_cell().geometric_dimension()
    # finite_element = element("P", domain.basix_cell(), degree, shape=(gdim,))
    # V = fem.functionspace(domain, finite_element)
    return 0

def preprocessing(example) -> mesh.Mesh:
    """Creates global coarse and fine scale grid"""

    from multi.preprocessing import create_rectangle_grid
    from multi.domain import StructuredQuadGrid

    # create coarse grid
    create_rectangle_grid(0., 10., 0., 1., num_cells=(example.nx, example.ny), recombine=True, out_file=example.coarse_grid.as_posix())

    # create unit cell grid
    create_rectangle_grid(0., 1. , 0., 1., num_cells=example.resolution, recombine=True, out_file=example.unit_cell_grid.as_posix())

    # initialize structured coarse grid
    coarse_domain, _, _ = gmshio.read_from_msh(example.coarse_grid.as_posix(), MPI.COMM_WORLD, gdim=2)
    coarse_grid = StructuredQuadGrid(coarse_domain)
    coarse_grid.fine_grid_method = [example.unit_cell_grid.as_posix()]
    # create fine grid
    coarse_grid.create_fine_grid(np.arange(coarse_grid.num_cells), example.fine_grid.as_posix(), cell_type="quad")

    # read fine grid from disk
    with XDMFFile(MPI.COMM_WORLD, example.fine_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")

    return domain

if __name__ == "__main__":
    # import argparse?
    # define those parameters once on doit task level
    resolution = 4
    degree = 1
    main(resolution, degree)
