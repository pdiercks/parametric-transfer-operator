import pathlib
import tempfile
import typing

import numpy as np
from dolfinx.io import XDMFFile, gmshio  # type: ignore
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.preprocessing import (
    create_meshtags,
    create_rectangle,
    create_voided_rectangle,
    merge_mshfiles,
)

from parageom.definitions import BeamData
from parageom.locmor import OversamplingConfig


def discretize_unit_cell(
    unit_length: float, radius: float, num_cells: int, output: str, gmsh_options: typing.Optional[dict] = None
) -> None:
    """Discretizes square domain with circular void.

    Args:
        unit_length: Unit length of the unit cell.
        radius: Radius of the void.
        num_cells: Number of cells per edge of the unit cell.
        output: Write .msh file.
        gmsh_options: Options for Gmsh.

    """
    xmin = 0.0
    xmax = unit_length
    ymin = 0.0
    ymax = unit_length

    create_voided_rectangle(
        xmin,
        xmax,
        ymin,
        ymax,
        radius=radius,
        num_cells=num_cells,
        recombine=True,
        cell_tags={'matrix': 1},
        facet_tags={'bottom': 11, 'left': 12, 'right': 13, 'top': 14, 'void': 15},
        out_file=output,
        options=gmsh_options,
    )


def create_structured_coarse_grid_v2(example, coarse_grid, active_cells, output: str):
    """Create a coarse grid partition of active cells of the global domain `coarse_grid`."""
    a = example.unit_length
    num_cells = active_cells.size

    left_most_cell = np.amin(active_cells)
    cell_vertices = coarse_grid.get_entities(0, left_most_cell)
    lower_left = cell_vertices[:1]
    xmin, ymin, _ = coarse_grid.get_entity_coordinates(0, lower_left)[0]

    xmax = xmin + a * num_cells
    ymin = 0.0
    ymax = 1.0 * a
    create_rectangle(xmin, xmax, ymin, ymax, num_cells=[num_cells, 1], recombine=True, out_file=output)


def create_structured_coarse_grid(example, typus: str, output: str):
    a = example.unit_length

    match typus:
        case 'global':
            num_cells = (example.nx, example.ny)
            xmin = 0.0
        case _:
            raise NotImplementedError

    xmax = xmin + a * num_cells[0]
    ymin = 0.0
    ymax = ymin + a * num_cells[1]
    create_rectangle(xmin, xmax, ymin, ymax, num_cells=num_cells, recombine=True, out_file=output)


def create_fine_scale_grid_v2(example, coarse_grid, active_cells, output: str):
    """Create fine grid discretization for `active_cells` of `coarse_grid`."""
    num_cells = active_cells.size

    subdomains = []
    to_be_merged = []

    facet_tags = []
    tag = 15
    for _ in range(num_cells):
        facet_tags.append({'void': tag})
        tag += 1

    offset = {2: 0, 1: 0}
    UNIT_LENGTH = example.unit_length
    for k, cell in enumerate(active_cells):
        subdomains.append(tempfile.NamedTemporaryFile(suffix='.msh'))
        to_be_merged.append(subdomains[k].name)
        gmsh_options = {'Mesh.ElementOrder': example.geom_deg, 'General.Verbosity': 0}

        cell_vertices = coarse_grid.get_entities(0, cell)
        lower_left = cell_vertices[:1]
        xc = coarse_grid.get_entity_coordinates(0, lower_left)
        xmin, ymin, zc = xc[0]
        xmax = xmin + UNIT_LENGTH
        ymax = ymin + UNIT_LENGTH
        radius = example.mu_bar

        create_voided_rectangle(
            xmin,
            xmax,
            ymin,
            ymax,
            z=zc,
            radius=radius,
            num_cells=example.num_intervals,
            recombine=True,
            cell_tags={'matrix': 1},
            facet_tags=facet_tags[k],
            out_file=to_be_merged[k],
            options=gmsh_options,
            tag_counter=offset,
        )

    merge_mshfiles(to_be_merged, output)
    for tmp in subdomains:
        tmp.close()


def create_fine_scale_grid(example, typus: str, output: str):
    """Create parent domain for `config`."""
    coarse_grid_msh = example.coarse_grid(typus).as_posix()
    coarse_domain = gmshio.read_from_msh(coarse_grid_msh, MPI.COMM_WORLD, gdim=example.gdim)[0]
    coarse_grid = StructuredQuadGrid(coarse_domain)
    num_cells = coarse_grid.num_cells

    subdomains = []
    to_be_merged = []

    facet_tags = []
    tag = 15
    for _ in range(num_cells):
        facet_tags.append({'void': tag})
        tag += 1

    offset = {2: 0, 1: 0}
    UNIT_LENGTH = example.unit_length
    for cell in range(num_cells):
        subdomains.append(tempfile.NamedTemporaryFile(suffix='.msh'))
        to_be_merged.append(subdomains[cell].name)
        gmsh_options = {'Mesh.ElementOrder': example.geom_deg}

        cell_vertices = coarse_grid.get_entities(0, cell)
        lower_left = cell_vertices[:1]
        xc = coarse_grid.get_entity_coordinates(0, lower_left)
        xmin, ymin, zc = xc[0]
        xmax = xmin + UNIT_LENGTH
        ymax = ymin + UNIT_LENGTH
        radius = example.mu_bar

        create_voided_rectangle(
            xmin,
            xmax,
            ymin,
            ymax,
            z=zc,
            radius=radius,
            num_cells=example.num_intervals,
            recombine=True,
            cell_tags={'matrix': 1},
            facet_tags=facet_tags[cell],
            out_file=to_be_merged[cell],
            options=gmsh_options,
            tag_counter=offset,
        )

    merge_mshfiles(to_be_merged, output)
    for tmp in subdomains:
        tmp.close()


def discretize_oversampling_domains(
    example: BeamData, struct_grid_gl: StructuredQuadGrid, osp_config: OversamplingConfig
):
    """Creates meshes for the oversampling domain and target subdomain.

    Args:
        example: The data class for the example problem.
        struct_grid_gl: Global coarse grid.
        osp_config: Configuration of this transfer problem.

    """
    cells_omega = osp_config.cells_omega
    cells_omega_in = osp_config.cells_omega_in

    # For the coarse grid there is no need to convert to xdmf
    # (since there are no meshtags)
    # Write to xdmf anyway for consistency?
    # In the future might want to load coarse grid in parallel
    # and let the automatic partitioning drive the local oversampling problems

    # create coarse grid partition of oversampling domain
    outstream = example.path_omega_coarse(osp_config.index)
    create_structured_coarse_grid_v2(example, struct_grid_gl, cells_omega, outstream.as_posix())
    coarse_omega = read_mesh(outstream, MPI.COMM_WORLD, kwargs={'gdim': example.gdim})[0]
    struct_grid_omega = StructuredQuadGrid(coarse_omega)
    assert struct_grid_omega.num_cells == cells_omega.size

    # create fine grid partition of oversampling domain
    path_omega_msh = tempfile.NamedTemporaryFile(suffix='.msh')
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega, path_omega_msh.name)
    omega, omega_ct, omega_ft = read_mesh(
        pathlib.Path(path_omega_msh.name), MPI.COMM_WORLD, kwargs={'gdim': example.gdim}
    )
    path_omega_msh.close()  # close and delete tmp msh file
    omega = RectangularDomain(omega, cell_tags=omega_ct, facet_tags=omega_ft)
    # create facets
    # facet tags for void interfaces start from 15 (see create_fine_scale_grid_v2)
    # i.e. 15 ... 24 for max number of cells

    facet_tag_definitions = {}
    for tag, key in zip([int(11), int(12), int(13)], ['bottom', 'left', 'right']):
        facet_tag_definitions[key] = (tag, omega.str_to_marker(key))

    # add tags for neumann boundary
    top_tag = None
    if osp_config.gamma_n is not None:
        top_tag = example.neumann_tag
        top_locator = osp_config.gamma_n
        facet_tag_definitions['top'] = (top_tag, top_locator)

    # update already existing facet tags
    # this will add tags for "top" boundary
    omega.facet_tags = create_meshtags(omega.grid, omega.tdim - 1, facet_tag_definitions, tags=omega.facet_tags)[0]

    assert omega.facet_tags.find(11).size == example.num_intervals * cells_omega.size  # bottom
    assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
    assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
    for itag in range(15, 15 + cells_omega.size):
        assert omega.facet_tags.find(itag).size == example.num_intervals * 4  # void

    path_omega = example.path_omega(args.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega.as_posix(), 'w') as xdmf:
        xdmf.write_mesh(omega.grid)
        xdmf.write_meshtags(omega.cell_tags, omega.grid.geometry)
        xdmf.write_meshtags(omega.facet_tags, omega.grid.geometry)

    # create fine grid partition of target subdomain
    path_omega_in_msh = tempfile.NamedTemporaryFile(suffix='.msh')
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega_in, path_omega_in_msh.name)
    omega_in, _, _ = read_mesh(pathlib.Path(path_omega_in_msh.name), MPI.COMM_WORLD, kwargs={'gdim': example.gdim})
    path_omega_in_msh.close()

    path_omega_in = example.path_omega_in(osp_config.index)
    with XDMFFile(MPI.COMM_WORLD, path_omega_in.as_posix(), 'w') as xdmf:
        xdmf.write_mesh(omega_in)


if __name__ == '__main__':
    import argparse
    import sys

    from parageom.locmor import oversampling_config_factory
    from parageom.tasks import example

    parser = argparse.ArgumentParser(description='Create grids for oversampling problems.')
    parser.add_argument(
        'k',
        type=int,
        help='The k-th oversampling problem.',
    )
    args = parser.parse_args(sys.argv[1:])

    # read global coarse grid
    coarse_grid_path = example.coarse_grid('global')
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': example.gdim})[0]
    struct_grid_gl = StructuredQuadGrid(coarse_domain)

    # oversampling configuration
    ospconf = oversampling_config_factory(args.k)

    discretize_oversampling_domains(example, struct_grid_gl, ospconf)
