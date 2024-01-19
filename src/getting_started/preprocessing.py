"""preprocessing module"""

from mpi4py import MPI
import numpy as np
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from basix.ufl import element
from multi.preprocessing import create_rectangle
from multi.domain import StructuredQuadGrid


def mark_subdomains(fine_grid: str, coarse_domain: mesh.Mesh) -> None:
    """Mark sudomains of the global fine scale grid"""

    # read fine grid from disk
    with XDMFFile(MPI.COMM_WORLD, fine_grid, "r") as fh:
        domain = fh.read_mesh(name="Grid")

    # create DG0 space on coarse grid
    dg_el = element("DG", coarse_domain.basix_cell(), 0, shape=())
    W = fem.functionspace(coarse_domain, dg_el)
    w = fem.Function(W)

    # +1 to stick to GMSH convention
    # cell tags should start at 1 instead of 0
    w.vector.array[:] = W.dofmap.list.flatten() + 1

    fe = element("DG", domain.basix_cell(), dg_el.degree(), shape=dg_el.value_shape())
    V = fem.functionspace(domain, fe)
    u = fem.Function(V)

    u.interpolate(w, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        u.function_space.mesh._cpp_object,
        u.function_space.element,
        w.function_space.mesh._cpp_object
        ))

    tdim = domain.topology.dim
    entities = V.dofmap.list.flatten()
    values = u.vector.array.astype(np.int32)
    cell_tags = mesh.meshtags(domain, tdim ,entities, values)
    cell_tags.name = "subdomains"

    domain.topology.create_connectivity(tdim-1, tdim)
    # overwrite xdmf file
    with XDMFFile(domain.comm, fine_grid, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags, domain.geometry)


def generate_meshes(example) -> None:
    """Creates grids for global FOM, ROM and localized ROM"""

    recombine = True
    options = {"Mesh.ElementOrder": example.fe_deg} # iso-parametric elements
    fine_grid_cell = "quad9"
    # Note: If mesh geometry and FE space are interpolated with same order
    # Lagrange polynomials, no additional interpolation is required when
    # writing to XDMFFile.

    # create coarse grid
    create_rectangle(0., example.length, 0., example.height, num_cells=(example.nx, example.ny), recombine=recombine, out_file=example.coarse_grid.as_posix(), options=options)

    # create unit cell grid
    create_rectangle(0., 1. , 0., 1., num_cells=example.resolution, recombine=recombine, out_file=example.unit_cell_grid.as_posix(), options=options)


    # initialize structured coarse grid
    coarse_domain, _, _ = gmshio.read_from_msh(example.coarse_grid.as_posix(), MPI.COMM_WORLD, gdim=2)
    coarse_grid = StructuredQuadGrid(coarse_domain)
    coarse_grid.fine_grid_method = [example.unit_cell_grid.as_posix()]

    # create global fine grid
    coarse_grid.create_fine_grid(np.arange(coarse_grid.num_cells), example.fine_grid.as_posix(), cell_type=fine_grid_cell, options=options)
    # create meshtags for fine scale grid
    mark_subdomains(example.fine_grid.as_posix(), coarse_domain)

    # ### oversampling domain per configuration
    configs = {
            # name: (xmin, xmax, ymin, ymax, num_cells_x, num_cells_y)
            "inner": (4.,  7., 0., 1., 3, 1),
            "left":  (0.,  2., 0., 1., 2, 1),
            "right": (8., 10., 0., 1., 2, 1),
            }
    for key, (xmin, xmax, ymin, ymax, nx, ny) in configs.items():

        create_rectangle(xmin, xmax, ymin, ymax, num_cells=(nx, ny), recombine=recombine, out_file=example.coarse_oversampling_grid(key).as_posix(), options=options)
        coarse_domain, _, _ = gmshio.read_from_msh(example.coarse_oversampling_grid(key).as_posix(), MPI.COMM_WORLD, gdim=2)
        coarse_grid = StructuredQuadGrid(coarse_domain)
        coarse_grid.fine_grid_method = [example.unit_cell_grid.as_posix()]
        coarse_grid.create_fine_grid(np.arange(coarse_grid.num_cells), example.fine_oversampling_grid(key).as_posix(), cell_type=fine_grid_cell, options=options)
        mark_subdomains(example.fine_oversampling_grid(key).as_posix(), coarse_domain)


if __name__ == "__main__":
    from definitions import BeamData
    beam = BeamData(name="beam")
    generate_meshes(beam)
