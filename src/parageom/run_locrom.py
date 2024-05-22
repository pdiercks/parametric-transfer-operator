from pathlib import Path

import numpy as np

import dolfinx as df

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.tools.random import new_rng
from pymor.operators.constructions import VectorOperator

from multi.dofmap import DofMap
from multi.io import BasesLoader


def main(args):
    from .tasks import example
    from .definitions import BeamProblem
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, discretize_subdomain_operators
    from .ei import interpolate_subdomain_operator
    from .locmor import reconstruct, assemble_system, EISubdomainOperatorWrapper

    # ### logger
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": example.log_run_locrom(
                args.nreal, args.method, args.distr
            )
        }
    )
    logger = getLogger(Path(__file__).stem, level="DEBUG")

    # ### Discretize global FOM
    global_parent_domain_path = example.global_parent_domain.as_posix()
    coarse_grid_path = example.coarse_grid("global").as_posix()
    # do not change interface tags; see src/parageom/preprocessing.py::create_parent_domain
    interface_tags = [i for i in range(15, 25)]
    global_auxp = discretize_auxiliary_problem(global_parent_domain_path, example.geom_deg, interface_tags, example.parameters["global"], coarse_grid=coarse_grid_path)
    trafo_d_gl = df.fem.Function(global_auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, global_auxp, trafo_d_gl)
    h1_product = fom.products["h1_0_semi"]

    # ### Discretize subdomain operators
    operator, rhs = discretize_subdomain_operators(example)

    # ### EI of subdomain operator
    mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator)
    restricted_op, _ = operator.restricted(magic_dofs)
    wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix)

    # ### Multiscale Problem
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    coarse_grid = beam_problem.coarse_grid

    # ### DofMap
    dofmap = DofMap(coarse_grid)

    # ### Reduced bases
    bases_folder = example.bases_path(args.nreal, args.method, args.distr)
    num_cells = example.nx * example.ny
    bases_loader = BasesLoader(bases_folder, num_cells)
    bases, num_max_modes = bases_loader.read_bases()
    num_max_modes_per_cell = np.amax(num_max_modes, axis=1)
    num_min_modes_per_cell = np.amin(num_max_modes, axis=1)
    max_modes = np.amax(num_max_modes_per_cell)
    min_modes = np.amin(num_min_modes_per_cell)
    logger.info(f"Global minimum number of modes per edge is: {min_modes}.")
    logger.info(f"Global maximum number of modes per edge is: {max_modes}.")

    # ### ROM Assembly and Error Analysis
    P = fom.parameters.space(example.mu_range)
    with new_rng(example.validation_set_seed):
        validation_set = P.sample_randomly(args.num_test)

    # better not create functions inside loops
    u_rb = df.fem.Function(fom.solution_space.V)
    u_loc = df.fem.Function(operator.source.V)

    max_errors = []
    max_relerrors = []

    # TODO set appropriate value for number of modes
    num_fine_scale_modes = list(range(0, max_modes + 1, 2))

    # Conversion of rhs to NumpyVectorSpace
    range_space = mops[0].range
    b = VectorOperator(range_space.from_numpy(
        rhs.as_range_array().to_numpy()
        ))

    for nmodes in num_fine_scale_modes:
        operator, rhs, local_bases = assemble_system(
                example, nmodes, dofmap, wrapped_op, b, bases, num_max_modes, fom.parameters
        )
        rom = StationaryModel(operator, rhs, name="locROM")

        fom_solutions = fom.solution_space.empty()
        rom_solutions = fom.solution_space.empty()

        err_norms = []
        for mu in validation_set:
            U_fom = fom.solve(mu)  # is this cached or computed everytime?
            fom_solutions.append(U_fom)
            U_rb_ = rom.solve(mu)

            reconstruct(U_rb_.to_numpy(), dofmap, local_bases, u_loc, u_rb)
            # copy seems necessary here
            # without it I get a PETSC ERROR (segmentation fault)
            U_rom = fom.solution_space.make_array([u_rb.vector.copy()])  # type: ignore
            rom_solutions.append(U_rom)

        err = fom_solutions - rom_solutions
        fom_norms = fom_solutions.norm(h1_product)
        err_norms = err.norm(h1_product)
        max_err = np.max(err_norms)
        logger.debug(f"{nmodes=}\tnum_dofs: {dofmap.num_dofs}\t{max_err=}")
        max_errors.append(max_err)
        max_relerrors.append(max_err / fom_norms[np.argmax(err_norms)])

        breakpoint()
        # TODO
        # debug rom; make sure that EI is the problem instead of something else ...
        # use full operator

    if args.output is not None:
        np.savetxt(
            args.output,
            np.vstack((num_fine_scale_modes, max_relerrors)).T,
            delimiter=",",
            header="modes, error",
        )

    if args.show:
        import matplotlib.pyplot as plt

        plt.title("ROM error relative to FOM")
        plt.semilogy(num_fine_scale_modes, max_relerrors, "k-o")
        plt.ylabel("Rel. error")
        plt.xlabel("Number of fine scale basis functions per edge")
        plt.show()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve the beam problem with parametrized geometry via localized MOR and compute the error relative to the FOM solution for a test set (sampled randomly)."
    )
    parser.add_argument("nreal", type=int, help="The number of the realization.")
    parser.add_argument(
        "method",
        type=str,
        help="The name of the training method.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument(
        "distr",
        type=str,
        help="The distribution used for sampling.",
        choices=("normal", "multivariate_normal"),
    )
    parser.add_argument(
        "num_test", type=int, help="Size of the test set used for validation."
    )
    parser.add_argument("--show", action="store_true", help="show error plot.")
    parser.add_argument(
        "--output", type=str, help="Path (.csv) to write relative error."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
