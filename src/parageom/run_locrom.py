"""Build and run localized ROM."""

from pathlib import Path
from collections import defaultdict

import numpy as np

import dolfinx as df
from dolfinx.common import Timer

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.constructions import VectorOperator
from pymor.models.basic import StationaryModel
from pymor.tools.random import new_rng

def main(args):
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, discretize_subdomain_operators

    from .ei import interpolate_subdomain_operator
    from .locmor import reconstruct, assemble_gfem_system, assemble_gfem_system_with_ei, EISubdomainOperatorWrapper
    from .dofmap_gfem import GFEMDofMap

    # ### logger
    set_defaults({"pymor.core.logger.getLogger.filename": example.log_run_locrom(args.nreal, args.method, args.distr, ei=args.ei)})
    if args.debug:
        loglevel = "DEBUG"
    else:
        loglevel = "INFO"
    logger = getLogger(Path(__file__).stem, level=loglevel)

    # ### Discretize global FOM
    global_parent_domain_path = example.parent_domain("global")
    coarse_grid_path = example.coarse_grid("global")
    # do not change interface tags; see src/parageom/preprocessing.py::create_parent_domain
    interface_tags = [i for i in range(15, 25)]
    global_auxp = discretize_auxiliary_problem(
        example,
        global_parent_domain_path.as_posix(),
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path.as_posix(),
    )
    trafo_d_gl = df.fem.Function(global_auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, global_auxp, trafo_d_gl)
    h1_product = fom.products["h1_0_semi"]

    # ### Discretize subdomain operators
    # NOTE
    # rhs_local is non-zero although this should only be the case for cell 0
    # this is handled during assembly
    # see `assemble_gfem_system` and `assemble_gfem_system_with_ei`
    operator_local, rhs_local = discretize_subdomain_operators(example)

    # ### EI of subdomain operator
    wrapped_op = None
    if args.ei:
        # FIXME
        # store data of deim somewhere
        with Timer("EI of subdomain operator") as t:
            mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator_local, design="uniform", ntrain=501, modes=None, atol=0., rtol=1e-12, method="method_of_snapshots")
            logger.info(f"EI of subdomain operator took {t.elapsed()[0]}.")
        m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
        logger.debug(f"{magic_dofs=}")
        restricted_op, _ = operator_local.restricted(m_dofs, padding=1e-8)
        wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix, magic_dofs, m_inv)

        # convert `rhs_local` to NumPy
        vector = rhs_local.as_range_array().to_numpy() # type: ignore
        rhs_va = mops[0].range.from_numpy(vector)
        rhs_local = VectorOperator(rhs_va)

    # ### Coarse grid of the global domain
    coarse_grid = global_auxp.coarse_grid # type: ignore

    # ### DofMap
    dofmap = GFEMDofMap(coarse_grid)

    # ### Reduced bases
    # 0: left, 1: transition, 2: inner, 3: transition, 4: right
    archetypes = []
    for cell in range(5):
        archetypes.append(np.load(example.local_basis_npy(args.nreal, args.method, args.distr, cell)))

    local_bases = []
    local_bases.append(archetypes[0].copy())
    local_bases.append(archetypes[1].copy())
    for _ in range(6):
        local_bases.append(archetypes[2].copy())
    local_bases.append(archetypes[3].copy())
    local_bases.append(archetypes[4].copy())
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
    P = fom.parameters.space(example.mu_range)
    with new_rng(example.validation_set_seed):
        validation_set = P.sample_randomly(args.num_test)

    # Functions to store FOM & ROM solution
    u_rb = df.fem.Function(fom.solution_space.V)
    u_loc = df.fem.Function(operator_local.source.V) # type: ignore

    Nmax = max_dofs_per_vert.max()
    ΔN = 10
    num_modes_per_vertex = list(range(Nmax // ΔN, Nmax + 1, Nmax // ΔN))
    logger.debug(f"{Nmax=}")
    logger.debug(f"{num_modes_per_vertex=}")

    l_char = example.l_char
    max_err = defaultdict(list)
    max_relerr = defaultdict(list)
    l2_err = []
    ndofs = []

    t_loop = Timer("Loop")
    t_loop.start()
    for nmodes in num_modes_per_vertex:

        # construct `dofs_per_vert` for current number of modes
        dofs_per_vert = max_dofs_per_vert.copy()
        dofs_per_vert[max_dofs_per_vert > nmodes] = nmodes
        # distribute dofs
        dofmap.distribute_dofs(dofs_per_vert)

        rom = None
        if args.ei:
            assert wrapped_op is not None
            with Timer("AssemblyEI") as t:
                operator, rhs, current_local_bases = assemble_gfem_system_with_ei(
                        dofmap, wrapped_op, rhs_local, local_bases, dofs_per_vert, max_dofs_per_vert, fom.parameters)
                logger.info(f"AssemblyEI took {t.elapsed()[0]}.")
            rom = StationaryModel(operator, rhs, name="locROM_with_ei")

        fom_solutions = fom.solution_space.empty()
        rom_solutions = fom.solution_space.empty()

        for mu in validation_set:
            U_fom = fom.solve(mu)
            fom_solutions.append(U_fom)

            if args.ei:
                assert rom is not None
                assert rom.name == "locROM_with_ei"
                with Timer("SolveEI") as t:
                    U_rb_ = rom.solve(mu)
                    logger.info(f"SolveEI took {t.elapsed()[0]}.")
            else:
                with Timer("Assembly") as t:
                    operator, rhs, current_local_bases = assemble_gfem_system(
                        dofmap,
                        operator_local,
                        rhs_local,
                        mu,
                        local_bases,
                        dofs_per_vert,
                        max_dofs_per_vert,
                    )
                    logger.info(f"{nmodes=}, \tAssembly took {t.elapsed()[0]}.")
                rom = StationaryModel(operator, rhs, name="locROM")

                with Timer("Solve") as t:
                    U_rb_ = rom.solve(mu)
                    logger.info(f"{nmodes=}, \tSolve took {t.elapsed()[0]}.")

            with Timer("reconstruction") as t:
                reconstruct(U_rb_.to_numpy(), dofmap, current_local_bases, u_loc, u_rb) # type: ignore
                logger.info(f"{nmodes=},\treconstruction took {t.elapsed()[0]}.")
            U_rom = fom.solution_space.make_array([u_rb.x.petsc_vec.copy()])  # type: ignore
            rom_solutions.append(U_rom)


        # absolute error
        err = fom_solutions - rom_solutions

        if args.debug:
            fom.visualize(fom_solutions[0], filename="output/run_locrom_fom.xdmf")
            fom.visualize(rom_solutions[0], filename="output/run_locrom_rom.xdmf")
            fom.visualize(err[0], filename="output/run_locrom_err.xdmf")

        # l2-mean error
        l2_mean = np.sum(l_char**2.0 * err.norm2(h1_product)) / len(err)

        # H1 norm
        err_norms = l_char * err.norm(h1_product)
        fom_norms = l_char * fom_solutions.norm(h1_product)
        rel_errn = err_norms / fom_norms
        logger.debug(f"{rel_errn=}")

        # Max norm (nodal absolute values)
        u_fom_vec = l_char * fom_solutions.amax()[1]
        e_vec = l_char * err.amax()[1]
        relerr_vec = e_vec / u_fom_vec

        summary = f"""Summary
        Modes:\t{nmodes}
        Ndofs:\t{dofmap.num_dofs}
        Test set size:\t{args.num_test}
        Max H1 relative error:\t{np.max(rel_errn)}
        Max max relative error:\t{np.max(relerr_vec)}
        l2-mean error:\t{l2_mean}
        """
        logger.info(summary)

        # ### Gather data
        if len(ndofs) < 1:
            # prepend value for nmodes=0
            ndofs.append(0)
            max_err["h1_semi"].append(np.max(fom_norms))
            max_err["max"].append(np.max(u_fom_vec))
            max_relerr["h1_semi"].append(1.0)
            max_relerr["max"].append(1.0)
            l2_err.append(np.sum(l_char**2.0 * fom_solutions.norm2(h1_product)) / len(fom_solutions))

        ndofs.append(dofmap.num_dofs)
        l2_err.append(l2_mean)
        max_err["h1_semi"].append(np.max(err_norms))
        max_relerr["h1_semi"].append(np.max(rel_errn))
        max_err["max"].append(np.max(e_vec))
        max_relerr["max"].append(np.max(relerr_vec))
    t_loop.stop()
    logger.info(f"Error analysis took {t_loop.elapsed()[0]}.")

    if args.output is not None:
        np.savez(
            args.output,
            ndofs=ndofs,
            max_err_h1_semi=max_err["h1_semi"],
            max_err_max=max_err["max"],
            max_relerr_h1_semi=max_relerr["h1_semi"],
            max_relerr_max=max_relerr["max"],
            l2_err=l2_err,
        )

    if args.show:
        import matplotlib.pyplot as plt

        plt.title("ROM error relative to FOM")
        plt.semilogy(ndofs, l2_err, "b-o", label="l2-mean")
        plt.semilogy(ndofs, max_relerr["h1_semi"], "k-o", label="H1")
        plt.ylabel("Error")
        plt.xlabel("Number of DOFs")
        plt.legend()
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
    parser.add_argument("num_test", type=int, help="Size of the test set used for validation.")
    parser.add_argument("--ei", action="store_true", help="Use empirical interpolation.")
    parser.add_argument("--show", action="store_true", help="Show error plot.")
    parser.add_argument("--debug", action="store_true", help="Set loglevel to DEBUG.")
    parser.add_argument("--output", type=str, help="Path (.npz) to write error.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
