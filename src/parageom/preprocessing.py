import typing
import tempfile
import numpy as np

from mpi4py import MPI
from dolfinx.io import gmshio
from multi.preprocessing import create_voided_rectangle, create_rectangle, merge_mshfiles
from multi.domain import StructuredQuadGrid


def discretize_unit_cell(
    unit_length, mu, num_cells: int, output: str, gmsh_options: typing.Optional[dict] = None
) -> None:
    """Discretizes square domain with circular void.

    Args:
        unit_length: Unit length of the unit cell.
        mu: parameter value, i.e. radius of the void.
        num_cells: Number of cells per edge of the unit cell.
        output: Write .msh file.

    """

    xmin = 0.0
    xmax = unit_length
    ymin = 0.0
    ymax = unit_length
    radius = mu.to_numpy().item()

    create_voided_rectangle(
        xmin,
        xmax,
        ymin,
        ymax,
        radius=radius,
        num_cells=num_cells,
        recombine=True,
        cell_tags={"matrix": 1},
        facet_tags={"bottom": 11, "left": 12, "right": 13, "top": 14, "void": 15},
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
    ymax = 1.0
    create_rectangle(
        xmin, xmax, ymin, ymax, num_cells=[num_cells, 1], recombine=True, out_file=output
    )


def create_structured_coarse_grid(example, typus: str, output: str):
    a = example.unit_length

    match typus:
        case "target":
            num_cells = (2, 1)
            xmin = 0.0
        case "left":
            num_cells = (3, 1)
            xmin = 0.0
        case "right":
            num_cells = (3, 1)
            xmin = 7 * a
        case "inner":
            num_cells = (4, 1)
            xmin = 3 * a
        case "global":
            num_cells = (example.nx, example.ny)
            xmin = 0.0
        case _:
            raise NotImplementedError

    xmax = xmin + a * num_cells[0]
    ymin = 0.0
    ymax = ymin + a * num_cells[1]
    create_rectangle(
        xmin, xmax, ymin, ymax, num_cells=num_cells, recombine=True, out_file=output
    )


def create_fine_scale_grid_v2(example, coarse_grid, active_cells, output: str):
    """Create fine grid discretization for `active_cells` of `coarse_grid`."""
    num_cells = active_cells.size

    subdomains = []
    to_be_merged = []

    facet_tags = []
    tag = 15
    for _ in range(num_cells):
        facet_tags.append({"void": tag})
        tag += 1

    offset = {2: 0, 1: 0}
    UNIT_LENGTH = example.unit_length
    for k, cell in enumerate(active_cells):
        subdomains.append(tempfile.NamedTemporaryFile(suffix=".msh"))
        to_be_merged.append(subdomains[k].name)
        gmsh_options = {"Mesh.ElementOrder": example.geom_deg, "General.Verbosity": 0}

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
            cell_tags={"matrix": 1},
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
    coarse_domain = gmshio.read_from_msh(coarse_grid_msh, MPI.COMM_WORLD, gdim=2)[0]
    coarse_grid = StructuredQuadGrid(coarse_domain)
    num_cells = coarse_grid.num_cells

    subdomains = []
    to_be_merged = []

    facet_tags = []
    tag = 15
    for _ in range(num_cells):
        facet_tags.append({"void": tag})
        tag += 1

    offset = {2: 0, 1: 0}
    UNIT_LENGTH = example.unit_length
    for cell in range(num_cells):
        subdomains.append(tempfile.NamedTemporaryFile(suffix=".msh"))
        to_be_merged.append(subdomains[cell].name)
        gmsh_options = {"Mesh.ElementOrder": example.geom_deg}

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
            cell_tags={"matrix": 1},
            facet_tags=facet_tags[cell],
            out_file=to_be_merged[cell],
            options=gmsh_options,
            tag_counter=offset,
        )

    merge_mshfiles(to_be_merged, output)
    for tmp in subdomains:
        tmp.close()


if __name__ == "__main__":
    import sys
    import argparse
    from parageom.tasks import example

    parser = argparse.ArgumentParser(
        description="Preprocessing for the parageom example."
    )
    parser.add_argument(
        "config",
        type=str,
        choices=("left", "inner", "right", "global", "target"),
        help="The configuration of the domain.",
    )
    parser.add_argument(
        "type",
        type=str,
        choices=("coarse", "fine"),
        help="Type of meshes that should be generated.",
    )
    parser.add_argument("--output", type=str, help="Write mesh to path if applicable.")
    args = parser.parse_args(sys.argv[1:])

    match args.type:
        case "coarse":
            assert args.output
            create_structured_coarse_grid(example, args.config, args.output)
        case "fine":
            create_fine_scale_grid(example, args.config, args.output)
