"""Build and run localized ROM."""

from pathlib import Path

import numpy as np

import dolfinx as df

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
# from pymor.tools.random import new_rng
# from pymor.operators.constructions import VectorOperator


def main(args):
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, discretize_subdomain_operators
    # from .ei import interpolate_subdomain_operator
    # from .locmor import reconstruct, assemble_system, assemble_system_with_ei, EISubdomainOperatorWrapper
    from .locmor import reconstruct, assemble_gfem_system
    from .dofmap_gfem import GFEMDofMap

    # ### logger
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": example.log_run_locrom(
                args.nreal, args.method, args.distr
            )
        }
    )
    logger = getLogger(Path(__file__).stem, level=10)

    # ### Discretize global FOM
    global_parent_domain_path = example.parent_domain("global")
    coarse_grid_path = example.coarse_grid("global")
    # do not change interface tags; see src/parageom/preprocessing.py::create_parent_domain
    interface_tags = [i for i in range(15, 25)]
    global_auxp = discretize_auxiliary_problem(global_parent_domain_path.as_posix(), example.geom_deg, interface_tags, example.parameters["global"], coarse_grid=coarse_grid_path.as_posix())
    trafo_d_gl = df.fem.Function(global_auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, global_auxp, trafo_d_gl)
    h1_product = fom.products["h1_0_semi"]

    # ### Discretize subdomain operators
    operator_local, rhs_local = discretize_subdomain_operators(example)

    # ### EI of subdomain operator
    # mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator_local)
    # restricted_op, _ = operator_local.restricted(magic_dofs)
    # wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix)

    # ### Coarse grid of the global domain
    coarse_grid = global_auxp.coarse_grid

    # ### DofMap
    dofmap = GFEMDofMap(coarse_grid)

    # ### Reduced bases
    # 0: left, 1: transition, 2: inner, 3: transition, 4: right
    archetypes = []
    for cell in range(5):
        archetypes.append(np.load(example.local_basis_npy(args.nreal, args.method, args.distr, cell)))

    local_bases = list((archetypes[2], ) * coarse_grid.num_cells)
    local_bases[0] = archetypes[0]
    local_bases[1] = archetypes[1]
    local_bases[-2] = archetypes[-2]
    local_bases[-1] = archetypes[-1]
    bases_length = [len(rb) for rb in local_bases]

    # ### Maximum number of modes per vertex
    max_dofs_per_vert = np.load(example.local_basis_dofs_per_vert(args.nreal, args.method, args.distr))
    # raise to number of cells in the coarse grid
    repetitions = [1, 1, coarse_grid.num_cells - len(archetypes) + 1, 1, 1]
    assert np.isclose(np.sum(repetitions), coarse_grid.num_cells)
    max_dofs_per_vert = np.repeat(max_dofs_per_vert, repetitions, axis=0)
    assert max_dofs_per_vert.shape == (coarse_grid.num_cells, 4)
    assert np.allclose(np.array(bases_length), np.sum(max_dofs_per_vert, axis=1))

    # ### ROM Assembly and Error Analysis
    # P = fom.parameters.space(example.mu_range)
    # with new_rng(example.validation_set_seed):
    #     validation_set = P.sample_randomly(args.num_test)
    validation_set = list()
    # mymu = fom.parameters.parse([0.12 * example.unit_length for _ in range(10)])
    mymu = fom.parameters.parse([0.2 * example.unit_length for _ in range(10)])
    # mymu = fom.parameters.parse([0.28 * example.unit_length for _ in range(10)])
    validation_set.append(mymu)

    # better not create functions inside loops
    u_rb = df.fem.Function(fom.solution_space.V)
    u_loc = df.fem.Function(operator_local.source.V)

    max_errors = []
    max_relerrors = []

    # TODO set appropriate value for number of modes
    # max_modes = 5
    # num_modes_per_vertex = list(range(1, max_modes + 1, 2))
    breakpoint()
    num_modes_per_vertex = [17, 21, 32]

    # Conversion of rhs to NumpyVectorSpace
    # range_space = mops[0].range
    # b = VectorOperator(range_space.from_numpy(
    #     rhs.as_range_array().to_numpy()
    #     ))

    for nmodes in num_modes_per_vertex:
        # operator, rhs, local_bases = assemble_system_with_ei(
        #         example, nmodes, dofmap, wrapped_op, b, bases, num_max_modes, fom.parameters
        # )
        # rom = StationaryModel(operator, rhs, name="locROM")

        fom_solutions = fom.solution_space.empty()
        rom_solutions = fom.solution_space.empty()

        err_norms = []
        for mu in validation_set:
            U_fom = fom.solve(mu)  # is this cached or computed everytime?
            fom_solutions.append(U_fom)

            # construct `dofs_per_vert` for current number of modes
            dofs_per_vert = max_dofs_per_vert.copy()
            dofs_per_vert[max_dofs_per_vert > nmodes] = nmodes
            # distribute dofs
            dofmap.distribute_dofs(dofs_per_vert)

            operator, rhs, current_local_bases = assemble_gfem_system(
                    dofmap, operator_local, rhs_local, mu, local_bases, dofs_per_vert, max_dofs_per_vert
                    )
            rom = StationaryModel(operator, rhs, name="locROM")
            breakpoint()
            U_rb_ = rom.solve(mu)

            reconstruct(U_rb_.to_numpy(), dofmap, current_local_bases, u_loc, u_rb)
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

    fom.visualize(fom_solutions, filename="ufom.xdmf")
    fom.visualize(rom_solutions, filename="urom.xdmf")
    breakpoint()

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
