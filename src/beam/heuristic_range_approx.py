import numpy as np
from pathlib import Path

from scipy.sparse.linalg import LinearOperator, eigsh
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.tools.random import new_rng
from pymor.parameters.base import Parameters, ParameterSpace


def lhs_design(beam, sampling_options):
    from .lhs import sample_lhs

    mu_name = sampling_options.get("name")
    ndim = sampling_options.get("ndim")
    samples = sampling_options.get("samples")
    criterion = sampling_options.get("criterion")
    random_state = sampling_options.get("random_state")

    param = Parameters({mu_name: ndim})
    parameter_space = ParameterSpace(param, beam.mu_range)
    training_set = sample_lhs(parameter_space, mu_name, samples=samples, criterion=criterion,
                              random_state=random_state)
    return training_set


def approximate_range(logger, beam, distribution, configuration, training_set, testing_set, lambda_min = None):
    """heuristic version of the range finder"""
    from .definitions import BeamProblem
    from .locmor import discretize_oversampling_problem

    assert distribution == "normal"

    # pid = os.getpid()
    # logger.info(f"{pid=},\tApproximating range of T for {mu=} using {distribution=}.\n")

    # Multiscale problem definition
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    cell_index = beam_problem.config_to_cell(configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    dirichlet = beam_problem.get_dirichlet(cell_index)
    active_edges = set(["bottom", "left", "right", "top"])

    # ttol = beam.rrf_ttol
    # ftol = beam.rrf_ftol
    num_testvecs = beam.rrf_num_testvecs
    # source_product = transfer_problem.source_product
    # range_product = beam.range_product

    # ### Initialize test sets
    test_sets = []
    mu_0 = testing_set[0]
    transfer_problem = discretize_oversampling_problem(beam, mu_0, configuration)
    R = transfer_problem.generate_random_boundary_data(num_testvecs, distribution)
    M = transfer_problem.solve(R)
    test_sets.append(M)
    # initialize source product
    source_product = transfer_problem.source_product
    del transfer_problem

    # ### Caculate min eigenvalue of source product matrix
    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()  # type: ignore

        def mvinv(v):
            return source_product.apply_inverse(
                source_product.range.from_numpy(v)  # type: ignore
            ).to_numpy()

        L = LinearOperator(
            (source_product.source.dim, source_product.range.dim), # type: ignore
            matvec=mv,  # type: ignore
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim), # type: ignore
            matvec=mvinv,  # type: ignore
        )
        lambda_min = eigsh(
            L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
        )[0]

    for mu in testing_set[1:]:
        transfer_problem = discretize_oversampling_problem(beam, mu, configuration)
        logger.debug(f"Transfer problem Reference count: {sys.getrefcount(transfer_problem)}.")
        R = transfer_problem.generate_random_boundary_data(num_testvecs, distribution)
        M = transfer_problem.solve(R)
        test_sets.append(M)
        del transfer_problem

    # subtract coarse scale part for each test set ...


def main(args):
    from .tasks import beam

    # TODO: add beam.log_heuristic_range_approx
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_heuristic_range_approx(
                args.distribution, args.configuration
            ).as_posix(),
        }
    )
    logger = getLogger(Path(__file__).stem, level="DEBUG")


    realizations = np.load(beam.realizations)
    # FIXME hardcoded to 1 realization
    test_seed, train_seed, rrf_seed = np.random.SeedSequence(realizations[0]).generate_state(3)

    # FIXME might need to add more samples here
    # how to select among these samples during the while loop?
    # ideally these would be selected in a greedy fashion
    train_options = beam.lhs_options[args.configuration]
    train_options["random_state"] = train_seed
    training_set = lhs_design(beam, train_options)

    test_options = beam.lhs_options[args.configuration]
    test_options["random_state"] = test_seed
    testing_set = lhs_design(beam, test_options)

    logger.info(
        "Starting heuristic range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
    )

    with new_rng(rrf_seed):
        basis = approximate_range(logger, beam, args.distribution, args.configuration,
                                  training_set, testing_set)



    # write fine scale basis
    # TODO beam.heuristic_fine_scale_edge_modes_npz

    # np.savez(
    #     beam.heuristic_fine_scale_edge_modes_npz(args.distribution, args.configuration),
    #     **basis,
    # )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Approximate the range of a parametric transfer operator using the heuristic rrf",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "distribution", type=str, help="The distribution to draw samples from."
    )
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="The max number of workers."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
