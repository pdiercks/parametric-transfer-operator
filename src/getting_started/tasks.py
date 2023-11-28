"""tasks for the getting started example"""

from .definitions import Example
ex = Example(name="beam")


def generate_meshes(example) -> None:
    """Creates global coarse and fine scale grid"""

    from mpi4py import MPI
    import numpy as np
    from dolfinx.io import gmshio
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


def task_preprocessing():
    """preprocessing: getting started"""
    return {
            "basename": f"preproc_{ex.name}",
            "actions": [(generate_meshes, [ex])],
            "targets": [ex.coarse_grid, ex.unit_cell_grid, ex.fine_grid],
            "clean": True,
            }
