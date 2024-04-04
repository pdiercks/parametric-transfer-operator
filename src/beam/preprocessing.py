"""preprocessing module"""

from mpi4py import MPI
import numpy as np
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from basix.ufl import element
from multi.preprocessing import create_rectangle, create_voided_rectangle
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

    fe = element(
        "DG", domain.basix_cell(), dg_el.degree, shape=dg_el.reference_value_shape
    )
    V = fem.functionspace(domain, fe)
    u = fem.Function(V)

    u.interpolate(
        w,
        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            u.function_space.mesh, u.function_space.element, w.function_space.mesh
        ),
    )

    tdim = domain.topology.dim
    entities = V.dofmap.list.flatten()
    values = u.vector.array.astype(np.int32)
    cell_tags = mesh.meshtags(domain, tdim, entities, values)
    cell_tags.name = "subdomains"

    domain.topology.create_connectivity(tdim - 1, tdim)
    # overwrite xdmf file
    with XDMFFile(domain.comm, fine_grid, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(cell_tags, domain.geometry)


def generate_meshes(example) -> None:
    """Creates grids for global FOM, ROM and localized ROM"""

    recombine = True
    options = {"Mesh.ElementOrder": example.geom_deg}  # iso-parametric elements
    if example.geom_deg == 1:
        fine_grid_cell = "quad"
    elif example.geom_deg == 2:
        fine_grid_cell = "quad9"
    else:
        raise NotImplementedError("Degree of geometry interpolation should be 1 or 2.")
    # Note: If mesh geometry and FE space are interpolated with same order
    # Lagrange polynomials, no additional interpolation is required when
    # writing to XDMFFile.

    # ### Global coarse scale grid
    create_rectangle(
        0.0,
        example.length,
        0.0,
        example.height,
        num_cells=(example.nx, example.ny),
        recombine=recombine,
        out_file=example.coarse_grid.as_posix(),
        options=options,
    )
    coarse_domain, _, _ = gmshio.read_from_msh(
        example.coarse_grid.as_posix(), MPI.COMM_WORLD, gdim=example.gdim
    )
    coarse_grid = StructuredQuadGrid(coarse_domain)

    # ### Unit cell grid
    create_voided_rectangle(
        0.0,
        1.0,
        0.0,
        1.0,
        num_cells=example.resolution,
        recombine=recombine,
        out_file=example.unit_cell_grid.as_posix(),
        options=options,
    )

    # ### Global fine scale grid
    coarse_grid.fine_grid_method = [example.unit_cell_grid.as_posix()]
    coarse_grid.create_fine_grid(
        np.arange(coarse_grid.num_cells),
        example.fine_grid.as_posix(),
        cell_type=fine_grid_cell,
        options=options,
    )
    mark_subdomains(example.fine_grid.as_posix(), coarse_domain)

    # ### Multiscale Problem
    from .definitions import BeamProblem

    problem = BeamProblem(example.coarse_grid.as_posix(), example.fine_grid.as_posix())

    # ### Creation of grids for oversampling problems
    cell_sets = problem.cell_sets_oversampling
    for key, cset in cell_sets.items():
        cells = np.array(list(cset), dtype=np.int32)
        os_domain, _, _, _ = mesh.create_submesh(coarse_domain, 2, cells)
        os_grid = StructuredQuadGrid(os_domain)
        os_grid.fine_grid_method = [example.unit_cell_grid.as_posix()]
        os_grid.create_fine_grid(
            np.arange(len(cells)),
            example.fine_oversampling_grid(key).as_posix(),
            cell_type=fine_grid_cell,
            options=options,
        )
        mark_subdomains(example.fine_oversampling_grid(key).as_posix(), os_domain)


if __name__ == "__main__":
    from .definitions import BeamData

    beam = BeamData(name="beam")
    generate_meshes(beam)
