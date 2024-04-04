from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx import fem
from basix.ufl import element
from pymor.bindings.fenicsx import FenicsxVectorSpace
from pymor.tools.random import new_rng
from multi.domain import StructuredQuadGrid, RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
import numpy as np


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
    dx = coarse_grid.get_entity_coordinates(0, np.array([cell_vertex], dtype=np.int32))
    omega.translate(dx)

    # ### Create Edge Spaces
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()

    # ### Unit cell problem
    ufl_element = fom.solution_space.V.ufl_element()
    fe = element(ufl_element.family_name, omega.grid.basix_cell(), ufl_element.degree, shape=ufl_element.reference_value_shape)
    V = fem.functionspace(omega.grid, fe)
    phases = LinearElasticMaterial(2, 20e3, 0.3) # material will not be important here
    problem = LinElaSubProblem(omega, V, phases=phases)
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    # ### Restrict snapshots to unit cell domain
    source = FenicsxVectorSpace(V)
    f = fem.Function(V)
    f_fom = fem.Function(fom.solution_space.V)

    test_set = source.empty()
    for uvec in snapshots.vectors:
        f_fom.vector.zeroEntries()
        f_fom.vector.axpy(1., uvec.real_part.impl)

        f.interpolate(f_fom, nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
            f.function_space.mesh,
            f.function_space.element,
            f_fom.function_space.mesh))

        U = source.make_array([f.vector.copy()])
        test_set.append(U)

    # ### Restrict test set to edges
    edge_functions = {}
    for edge, dofs in problem.V_to_L.items():
        edge_functions[edge] = test_set.dofs(dofs)

    cell_to_config = {0: "left", 4: "inner", 9: "right"}
    config = cell_to_config[args.subdomain_id]
    np.savez(beam.fom_test_set(config), **edge_functions)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Generates test data for the displacement field of a unit cell from FOM solutions. The FOM solutions are restricted to the given unit cell index (i.e. subdomain index).", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("num_solves", type=int, help="The number of FOM solutions to compute.")
    parser.add_argument("subdomain_id", type=int, help="The subdomain index.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
