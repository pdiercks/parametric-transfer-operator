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

    parameter_space = ParameterSpace(example.parameters[args.config], example.mu_range)
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
    epsilon_star = example.epsilon_star_projerr

    if args.method == "hapod":
        from .hapod import adaptive_rrf_normal

        snapshots = transfer.range.empty()
        spectral_basis_sizes = list()

        Nin = transfer.rhs.dofs.size
        epsilon_alpha = np.sqrt(Nin) * np.sqrt(1 - example.omega**2.0) * epsilon_star
        # total number of input vectors is at most Nin * ntrain
        epsilon_pod = np.sqrt(Nin * ntrain) * example.omega * epsilon_star
        # but as usually much less vectors than Nin are computed per transfer operator
        # however epsilon_pod can simply be computed after number of snapshots is known

        # ε_α = np.sqrt(Nin) ... <-- cardinality of snapshot set
        # l2-mean should be computed over set of size Nin then ...

        # but since Nin is 

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
                    num_testvecs=Nin,
                    # num_testvecs=20,
                    l2_err=epsilon_alpha,
                    # sampling_options={"scale": 0.1},
                )
                logger.info(f"\nSpectral Basis length: {len(rb)}.")
                spectral_basis_sizes.append(len(rb))
                snapshots.append(rb) # type: ignore
        logger.info(
            f"Average length of spectral basis: {np.average(spectral_basis_sizes)}."
        )
        basis = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)[0]  # type: ignore

    elif args.method == "heuristic":
        from .heuristic import heuristic_range_finder

        l2_err = epsilon_star / example.l_char

        with new_rng(seed_seqs_rrf[0]):
            spectral_basis = heuristic_range_finder(
                logger,
                transfer,
                training_set,
                testing_set,
                error_tol=example.rrf_ttol / example.l_char,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=example.rrf_num_testvecs,
                l2_err=l2_err,
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

    # Definition of test set (μ) and test data (g)
    test_set = parameter_space.sample_randomly(ntrain)
    test_data = transfer.range.empty(reserve=len(test_set))

    # use ntrain and Nin to define test data, but with different seed?

    logger.info(f"Computing test set of size {len(test_set)}...")
    with new_rng(example.projerr_seed):
        for mu in test_set:
            transfer.assemble_operator(mu)
            # g = transfer.generate_random_boundary_data(1, args.distr, {"scale": 0.1})
            g = transfer.generate_random_boundary_data(Nin, args.distr)
            test_data.append(transfer.solve(g))

    aerrs = defaultdict(list)
    rerrs = defaultdict(list)
    l2errs = defaultdict(list)

    def compute_norm(U, key, value):
        lc = example.l_char

        if key == "max":
            return lc * U.amax()[1]
        else:
            assert key in (transfer.range_product.name, "euclidean")
            return lc * U.norm(value)

    products = {transfer.range_product.name: transfer.range_product, "euclidean": None, "max": False}
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
        error = test_data - U_proj  # type: ignore
        for k, v in products.items():
            error_norm = compute_norm(error, k, v)
            if np.all(error_norm == 0.0):
                # ensure to return 0 here even when the norm of U is zero
                rel_err = error_norm
            else:
                rel_err = error_norm / test_norms[k]
            l2_err = np.sum(error_norm**2.0) / len(test_data)

            aerrs[k].append(np.max(error_norm))
            rerrs[k].append(np.max(rel_err))
            l2errs[k].append(l2_err)

    breakpoint()
    print(np.min(l2errs["energy"]))
    # FIXME? when using the energy product min l2err is not below epsilon_star ** 2
    # during the training (rrf) I compute l2-mean over set of size Nin (=84 for inner)
    # but the projection error is computed over set of size 500
    if args.output is not None:
        np.savez(
            args.output,
            rerr_h1_semi=rerrs[transfer.range_product.name],
            rerr_euclidean=rerrs["euclidean"],
            rerr_max=rerrs["max"],
            aerr_h1_semi=aerrs[transfer.range_product.name],
            aerr_euclidean=aerrs["euclidean"],
            aerr_max=aerrs["max"],
            l2err_h1_semi=l2errs[transfer.range_product.name],
            l2err_euclidean=l2errs["euclidean"],
            l2err_max=l2errs["max"],
        )


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
