"""compute projection error to assess quality of the basis"""

from mpi4py import MPI
import dolfinx as df

from multi.io import read_mesh, BasesLoader, select_modes
from multi.interpolation import make_mapping
from multi.dofmap import DofMap
from multi.projection import compute_absolute_proj_errors
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def main(args):
    from .tasks import example
    from .definitions import BeamProblem
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom

    # get FOM solution
    # check pointing not yt available, repeat steps from run_locrom.py
    # ### Discretize global FOM
    global_parent_domain_path = example.global_parent_domain.as_posix()
    coarse_grid_path = example.coarse_grid("global").as_posix()
    # do not change interface tags; see src/parageom/preprocessing.py::create_parent_domain
    interface_tags = [i for i in range(15, 25)]
    global_auxp = discretize_auxiliary_problem(
        global_parent_domain_path,
        example.geom_deg,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    trafo_d_gl = df.fem.Function(global_auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, global_auxp, trafo_d_gl)

    # U_ref = fom.solve(mu_ref)
    # U = fom.solve(mu)

    # extract u for single cell
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    coarse_grid = beam_problem.coarse_grid
    delta_x = beam_problem.get_xmin_omega_in(args.cell)

    unit_cell, _, _ = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, gdim=2)
    x_geom = unit_cell.geometry.x
    x_geom += delta_x
    V = df.fem.functionspace(unit_cell, fom.solution_space.V.ufl_element())

    dofs = make_mapping(V, fom.solution_space.V)

    # read reduced basis from file
    nreal = 0
    method = "hapod"
    distr = "normal"
    bases_folder = example.bases_path(nreal, method, distr)
    num_cells = example.nx * example.ny
    bases_loader = BasesLoader(bases_folder, num_cells)
    bases, num_max_modes = bases_loader.read_bases()

    dofmap = DofMap(coarse_grid)
    dofs_per_vertex = 2
    dofs_per_face = 0
    num_modes = 12

    dofs_per_edge = num_max_modes.copy()
    dofs_per_edge[num_max_modes > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vertex, dofs_per_edge, dofs_per_face)
    local_basis = select_modes(
        bases[args.cell], num_max_modes[args.cell], dofs_per_edge[args.cell]
    )

    # ### wrap as pymor objects
    source = FenicsxVectorSpace(V)
    basis = source.from_numpy(local_basis)

    inner_product = InnerProduct(V, "h1")
    product_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(product_mat, V, V)

    # compute projection error
    test_set = []
    test_set.append(fom.parameters.parse([0.12 * example.unit_length for _ in range(10)]))
    test_set.append(fom.parameters.parse([0.2 * example.unit_length for _ in range(10)]))
    test_set.append(fom.parameters.parse([0.27 * example.unit_length for _ in range(10)]))

    for mu in test_set:
        U_fom = fom.solve(mu)
        u_local = U_fom.dofs(dofs)
        U = source.from_numpy(u_local)
        error = compute_absolute_proj_errors(U, basis, product=product, orthonormal=False)
        print(f"min error = {min(error)} for mu = {mu.to_numpy()[args.cell]} and cell {args.cell}")

    # confirm that current basis yields good results if mu=mu_bar
    # but fails for other values of mu


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cell", type=int, help="The coarse grid cell to use.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
