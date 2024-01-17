from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
from basix.ufl import element
from pymor.bindings.fenicsx import FenicsxVectorSpace
from pymor.tools.random import new_rng
from multi.domain import StructuredQuadGrid, RectangularSubdomain


def main(args):
    from .tasks import beam
    from .fom import discretize_fom

    # ### discretize FOM
    fom = discretize_fom(beam)
    parameter_space = fom.parameters.space(beam.mu_range)
    with new_rng(1990):
        parameter_set = parameter_space.sample_randomly(args.num_solves)

    # ### compute FOM solutions
    snapshots = fom.solution_space.empty()
    for mu in parameter_set:
        snapshots.append(fom.solve(mu))

    # ### read global coarse grid
    with beam.coarse_grid.open("r") as fh:
        coarse_grid, _, _ = gmshio.read_from_msh(fh.name, MPI.COMM_SELF, gdim=2)
    coarse_grid = StructuredQuadGrid(coarse_grid)

    # ### read unit cell grid (subdomain grid)
    with beam.unit_cell_grid.open("r") as fh:
        unit_cell_domain, _, _ = gmshio.read_from_msh(fh.name, MPI.COMM_SELF, gdim=2)
    omega = RectangularSubdomain(args.subdomain_id, unit_cell_domain)

    # ### Translate unit cell domain
    cell_vertex = coarse_grid.get_entities(0, args.subdomain_id)[0]
    dx = coarse_grid.get_entity_coordinates(0, cell_vertex)
    omega.translate(dx)

    # ### Restrict snapshots to unit cell domain
    ufl_element = fom.solution_space.V.ufl_element()
    fe = element(ufl_element.family_name, omega.grid.basix_cell(), ufl_element.degree(), shape=ufl_element.value_shape())
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)
    f = fem.Function(V)
    f_fom = fem.Function(fom.solution_space.V)

    test_set = source.empty()
    for uvec in snapshots.vectors:
        f_fom.vector.zeroEntries()
        f_fom.vector.axpy(1., uvec.real_part.impl)

        f.interpolate(f_fom, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            f.function_space.mesh._cpp_object,
            f.function_space.element,
            f_fom.function_space.mesh._cpp_object))

        U = source.make_array([f.vector.copy()])
        test_set.append(U)
    np.save(beam.fom_test_set(args.subdomain_id), test_set.to_numpy())


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generates test data for the displacement field of a unit cell from FOM solutions. The FOM solutions are restricted to the given unit cell index (i.e. subdomain index).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("num_solves", type=int, help="The number of FOM solutions to compute.")
    parser.add_argument("subdomain_id", type=int, help="The subdomain index.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
