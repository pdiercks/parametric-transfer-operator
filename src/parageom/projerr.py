"""compute projection error to assess quality of the basis"""

import numpy as np

from multi.projection import project_array

from pymor.algorithms.pod import pod
from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults
from pymor.tools.random import new_rng
from pymor.parameters.base import ParameterSpace


def main(args):
    from .tasks import example
    from .lhs import sample_lhs
    from .locmor import discretize_transfer_problem

    logfilename = example.log_projerr(
        args.nreal, args.method, args.distr, args.config
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("projerr", level="INFO")

    transfer, _ = discretize_transfer_problem(example, args.config)

    ntrain = example.ntrain(args.config)
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # ### Generate training and testing seed for each configuration
    training_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.training_set_seed).generate_state(
            len(example.configurations)
        ),
    ):
        training_seeds[cfg] = rndseed
    testing_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.testing_set_seed).generate_state(
            len(example.configurations)
        ),
    ):
        testing_seeds[cfg] = rndseed

    parameter_space = ParameterSpace(
        example.parameters[args.config], example.mu_range
    )
    parameter_name = list(example.parameters[args.config].keys())[0]
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=training_seeds[args.config],
    )
    testing_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=testing_seeds[args.config],
    )

    # ### Read basis and wrap as pymor object
    logger.info(f"Computing spectral basis with method {args.method} ...")
    basis = None

    # TODO
    # parametrize omega?
    epsilon_star = example.epsilon_star
    omega = example.omega

    if args.method == "hapod":
        from .hapod import adaptive_rrf_normal

        snapshots = transfer.range.empty()
        spectral_basis_sizes = list()
        epsilon_star = epsilon_star[args.method]
        epsilon_alpha = np.sqrt(example.rrf_num_testvecs) * np.sqrt(1 - omega**2.) * epsilon_star

        for mu, seed_seq in zip(training_set, seed_seqs_rrf):
            with new_rng(seed_seq):
                transfer.assemble_operator(mu)
                rb = adaptive_rrf_normal(
                    logger,
                    transfer,
                    error_tol=example.rrf_ttol,
                    failure_tolerance=example.rrf_ftol,
                    num_testvecs=example.rrf_num_testvecs,
                    l2_err=epsilon_alpha,
                    sampling_options={"scale": 0.1},
                )
                logger.info(f"\nSpectral Basis length: {len(rb)}.")
                spectral_basis_sizes.append(len(rb))
                snapshots.append(rb)
        logger.info(
            f"Average length of spectral basis: {np.average(spectral_basis_sizes)}."
        )
        epsilon = np.sqrt(len(snapshots)) * epsilon_star
        basis = pod(snapshots, product=transfer.range_product, l2_err=epsilon)[0]  # type: ignore

    elif args.method == "heuristic":
        from .heuristic import heuristic_range_finder
        epsilon_star = epsilon_star[args.method]

        with new_rng(seed_seqs_rrf[0]):
            spectral_basis, _ = heuristic_range_finder(
                logger,
                transfer,
                training_set,
                testing_set,
                error_tol=example.rrf_ttol,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=example.rrf_num_testvecs,
                l2_err=epsilon_star,
                sampling_options={"scale": 0.1},
            )
        basis = spectral_basis
    else:
        raise NotImplementedError

    orthonormal = np.allclose(
        basis.gramian(transfer.range_product), np.eye(len(basis)), atol=1e-5
    )
    if not orthonormal:
        raise ValueError("Basis is not orthonormal wrt range product.")

    # Definition of validation set
    # make sure that this is always the same set of parameters
    # and also same set of boundary data
    # but different from μ and g used in the training
    # purely random testing would be better ??
    # test_set = sample_lhs(
    #     parameter_space,
    #     name=parameter_name,
    #     samples=30,
    #     criterion="center",
    #     random_state=example.projerr_seed,
    # )

    test_set = parameter_space.sample_randomly(50)
    test_data = transfer.range.empty(reserve=len(test_set))

    logger.info(f"Computing test set of size {len(test_set)}...")
    with new_rng(example.projerr_seed):
        for mu in test_set:
            transfer.assemble_operator(mu)
            g = transfer.generate_random_boundary_data(1, args.distr, {"scale": 0.1})
            test_data.append(transfer.solve(g))

    aerrs = []
    rerrs = []
    l2errs = []
    u_norm = test_data.norm(transfer.range_product)  # norm of each test vector

    logger.info("Computing projection error ...")
    for N in range(len(basis) + 1):
        U_proj = project_array(
            test_data,
            basis[:N],
            product=transfer.range_product,
            orthonormal=orthonormal,
        )
        err = test_data - U_proj # type: ignore
        errn = err.norm(transfer.range_product)  # absolute projection error
        if np.all(errn == 0.0):
            # ensure to return 0 here even when the norm of U is zero
            rel_err = errn
        else:
            rel_err = errn / u_norm
        l2_err = np.sum((err).norm2(transfer.range_product)) / len(test_data)

        aerrs.append(np.max(errn))
        rerrs.append(np.max(rel_err))
        l2errs.append(l2_err)

    rerr = np.array(rerrs)
    aerr = np.array(aerrs)
    l2err = np.array(l2errs)
    if args.output is not None:
        np.savez(args.output, rerr=rerr, aerr=aerr, l2err=l2err)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument(
        "distr", type=str, help="Distribution used for random sampling."
    )
    parser.add_argument("config", type=str, help="Configuration / Archetype.")
    parser.add_argument(
        "--output",
        type=str,
        help="Write absolute and relative projection error to file.",
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
