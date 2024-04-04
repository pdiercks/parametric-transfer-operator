def discretize_unit_cell(mu, num_cells: int, output: str) -> None:
    """Discretizes square domain with circular void.

    Args:
        mu: parameter value, i.e. radius of the void.
        num_cells: Number of cells per edge of the unit cell.
        output: Write .msh file.

    """
    from multi.preprocessing import create_voided_rectangle

    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
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
    )


def main():
    from pymor.parameters.base import Parameters

    parameters = Parameters({"R": 1})
    mu_bar = parameters.parse([0.2])  # TODO use center of mu_range
    reference_unit_cell = "./reference_unit_cell.msh"
    num_cells = 12
    discretize_unit_cell(mu_bar, num_cells, reference_unit_cell)


if __name__ == "__main__":
    # Usage:
    # import functions implemented here in the dodo.py?

    # for all mu in training set
    # mu --> d(μ) --> transformed mesh for physical simulation (training)
    # (training) --> reduced basis and EI --> localized ROM

    # TODO creation of meshes for oversampling
    # TODO creation of meshes for FOM (ROM validation)
    main()
