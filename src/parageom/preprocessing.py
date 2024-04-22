import typing

UNIT_LENGTH = 1.0


def discretize_unit_cell(
    mu, num_cells: int, output: str, gmsh_options: typing.Optional[dict] = None
) -> None:
    """Discretizes square domain with circular void.

    Args:
        mu: parameter value, i.e. radius of the void.
        num_cells: Number of cells per edge of the unit cell.
        output: Write .msh file.

    """
    from multi.preprocessing import create_voided_rectangle

    xmin = 0.0
    xmax = UNIT_LENGTH
    ymin = 0.0
    ymax = UNIT_LENGTH
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


def create_structured_coarse_grid(config: str, output: str):
    from .tasks import example
    from multi.preprocessing import create_rectangle

    match config:
        case "left":
            num_cells = (2, 1)
            xmin = 0.0
        case "right":
            num_cells = (2, 1)
            xmin = 8.0
        case "inner":
            num_cells = (3, 1)
            xmin = 3.0
        case "global":
            num_cells = (example.nx, example.ny)
            xmin = 0.0
        case _:
            raise NotImplementedError

    xmax = xmin + UNIT_LENGTH * num_cells[0]
    ymin = 0.0
    ymax = ymin + UNIT_LENGTH * num_cells[1]
    create_rectangle(
        xmin, xmax, ymin, ymax, num_cells=num_cells, recombine=True, out_file=output
    )


def create_parent_domain(config: str, output: str):
    import tempfile

    from mpi4py import MPI
    from dolfinx.io import gmshio

    from multi.domain import StructuredQuadGrid
    from multi.preprocessing import create_voided_rectangle, merge_mshfiles

    from .tasks import example

    coarse_grid_msh = example.coarse_grid(config).as_posix()
    coarse_domain = gmshio.read_from_msh(coarse_grid_msh, MPI.COMM_WORLD, gdim=2)[0]
    coarse_grid = StructuredQuadGrid(coarse_domain)
    num_cells = coarse_grid.num_cells

    subdomains = []
    to_be_merged = []
    facet_tags = [
        {"void": 15},
        {"void": 16},
        {"void": 17},
        {"void": 18},
        {"void": 19},
        {"void": 20},
        {"void": 21},
        {"void": 22},
        {"void": 23},
        {"void": 24},
    ]

    offset = {2: 0, 1: 0}
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

        create_voided_rectangle(
            xmin,
            xmax,
            ymin,
            ymax,
            z=zc,
            radius=example.mu_bar,
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


def create_physical_domain(config: str):
    import pathlib
    import tempfile

    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.io import gmshio
    from dolfinx.io.utils import XDMFFile

    import numpy as np

    from pymor.core.pickle import load
    from multi.domain import StructuredQuadGrid

    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    with open(example.training_set(config), "rb") as fh:
        training_data = load(fh)
    training_set = training_data["training_set"]

    reference_cell = example.parent_unit_cell.as_posix()
    aux = discretize_auxiliary_problem(
        reference_cell, example.geom_deg, example.parameters["subdomain"]
    )
    d = fem.Function(aux.problem.V, name="transformation displacement")

    coarse_grid_msh = example.coarse_grid(config).as_posix()
    coarse_domain = gmshio.read_from_msh(coarse_grid_msh, MPI.COMM_WORLD, gdim=2)[0]
    coarse_grid = StructuredQuadGrid(coarse_domain)

    for k, mu in enumerate(training_set):
        subdomains = []  # list of msh files

        for mu_i in mu.to_numpy():
            # read reference mesh
            local_mu = aux.parameters.parse([mu_i])
            d.x.array[:] = 0.0
            aux.solve(d, local_mu)
            d.x.scatter_forward()

            # translation is done internally by StructuredQuadGrid.create_fine_grid
            # therefore only need to apply transformation displacement

            # read parent unit cell via gmshio
            # (cannot use meshio because of dof layout, cell ordering)
            parent_subdomain, _, _ = gmshio.read_from_msh(
                reference_cell, MPI.COMM_WORLD, gdim=2
            )
            x_subdomain = parent_subdomain.geometry.x
            disp = np.pad(
                d.x.array.reshape(x_subdomain.shape[0], -1), pad_width=[(0, 0), (0, 1)]
            )
            x_subdomain += disp

            # write physical subdomain mesh to XDMF
            tmp_xdmf = tempfile.NamedTemporaryFile(suffix=".xdmf", delete=False)
            xdmf = XDMFFile(parent_subdomain.comm, tmp_xdmf.name, "w")
            xdmf.write_mesh(parent_subdomain)
            xdmf.close()

            # FIXME
            # Writing facet tags has issues.
            # The XDMFFile will not be readable by meshio. (XDMF Reader only supports one grid)

            subdomains.append(tmp_xdmf.name)

        oversampling_msh = example.oversampling_domain(config, k).as_posix()
        coarse_grid.fine_grid_method = subdomains
        coarse_grid.create_fine_grid(
            np.arange(len(subdomains)),
            output=oversampling_msh,
            cell_type=example.cell_type,
        )

        # delete tmp files
        for tmp in subdomains:
            pathlib.Path(tmp).unlink()
        subdomains.clear()


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing for the parageom example."
    )
    parser.add_argument(
        "config",
        type=str,
        choices=("left", "inner", "right", "global"),
        help="The configuration of the domain.",
    )
    parser.add_argument(
        "type",
        type=str,
        choices=("coarse", "oversampling", "parent"),
        help="Type of meshes that should be generated.",
    )
    parser.add_argument("--output", type=str, help="Write mesh to path if applicable.")
    args = parser.parse_args(sys.argv[1:])

    match args.type:
        case "coarse":
            # generate coarse grid meshes of oversampling domain
            assert args.output
            create_structured_coarse_grid(args.config, args.output)
        case "oversampling":
            # generate fine grid meshes of oversampling domains
            # TODO creation of meshes for oversampling
            create_physical_domain(args.config)
        case "parent":
            create_parent_domain(args.config, args.output)

    # TODO creation of global meshes for FOM (ROM validation)
