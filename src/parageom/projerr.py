"""compute projection error to assess quality of the basis"""

from collections import defaultdict
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

    if args.method == "hapod":
        from .hapod import adaptive_rrf_normal

        snapshots = transfer.range.empty()
        spectral_basis_sizes = list()

        epsilon_star = example.epsilon_star[args.method]
        Nin = transfer.rhs.dofs.size
        epsilon_alpha = np.sqrt(Nin) * np.sqrt(1 - example.omega**2.) * epsilon_star
        epsilon_pod = epsilon_star * np.sqrt(Nin * ntrain)

        # scaling
        epsilon_alpha /= example.l_char
        epsilon_pod /= example.l_char

        for mu, seed_seq in zip(training_set, seed_seqs_rrf):
            with new_rng(seed_seq):
                transfer.assemble_operator(mu)
                rb = adaptive_rrf_normal(
                    logger,
                    transfer,
                    error_tol=example.rrf_ttol / example.l_char,
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
        basis = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)[0]  # type: ignore

    elif args.method == "heuristic":
        from .heuristic import heuristic_range_finder
        epsilon_star = example.epsilon_star[args.method] / example.l_char

        with new_rng(seed_seqs_rrf[0]):
            spectral_basis, _ = heuristic_range_finder(
                logger,
                transfer,
                training_set,
                testing_set,
                error_tol=example.rrf_ttol / example.l_char,
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

    aerrs = defaultdict(list)
    rerrs = defaultdict(list)
    l2errs = defaultdict(list)

    def compute_norm(U, key, value):
        lc = example.l_char

        if key == "max":
            return lc * U.amax()[1]
        else:
            assert key in ("h1-semi", "euclidean")
            return lc * U.norm(value)

    products = {"h1-semi": transfer.range_product, "euclidean": None, "max": False}
    test_norms = {}
    for k, v in products.items():
        test_norms[k] = compute_norm(test_data, k, v)

    logger.info("Computing projection error ...")
    for N in range(len(basis) + 1):
        U_proj = project_array(
            test_data,
            basis[:N],
            product=transfer.range_product,
            orthonormal=orthonormal,
        )
        error = test_data - U_proj # type: ignore
        for k, v in products.items():
            error_norm = compute_norm(error, k, v)
            if np.all(error_norm == 0.0):
                # ensure to return 0 here even when the norm of U is zero
                rel_err = error_norm
            else:
                rel_err = error_norm / test_norms[k]
            l2_err = np.sum(error_norm ** 2.) / len(test_data)

            aerrs[k].append(np.max(error_norm))
            rerrs[k].append(np.max(rel_err))
            l2errs[k].append(l2_err)

    # data = {"aerr": aerrs, "rerr": rerrs, "l2err": l2errs, "test_norms": test_norms}
    # TODO:
    # write out error in euclidean norm
    # write out max value of test_data, U_proj and error

    # Summary
    # epsilon_star = 0.1
    # aerrs['max'][-1] = 0.0634 (in mm, because norm is scaled with lc)
    # rerrs['max'][-1] = 0.0613
    # aerrs['h1-semi'][-1] = 0.198
    # l2errs['h1-semi'][-1] = 0.0078 (<1e-2=epsilon_star**2)

    # TODO:
    # run everything with epsilon_star = 0.1
    # and compare actual error in run_locrom ...

    # Summary
    # epsilon_star = 0.01
    # aerrs['max'][-1] = 0.0035 (in mm, because norm is scaled with lc)
    # rerrs['max'][-1] = 0.00523
    # aerrs['h1-semi'][-1] = 0.018
    # l2errs['h1-semi'][-1] = 5.7e-05 (<1e-4=epsilon_star**2)

    # see which ROM error epsilon_star=0.001 yields (run_locrom)
    # if max nodal ROM error is well below 1e-3 might rather use epsilon_star=0.01
    if args.output is not None:
        np.savez(args.output, rerr=rerrs["h1-semi"], aerr=aerrs["h1-semi"], l2err=l2errs["h1-semi"])


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
