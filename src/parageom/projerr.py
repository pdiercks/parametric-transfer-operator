"""compute projection error to assess quality of the basis"""

from collections import defaultdict

from dolfinx.io import XDMFFile # type: ignore
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import project_array
import numpy as np
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.parameters.base import ParameterSpace
from pymor.tools.random import new_rng


def main(args):
    from parageom.tasks import example
    from parageom.lhs import sample_lhs
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    if args.k in (0, 1, 2):
        raise NotImplementedError("Choose an oversampling problem without Neumann data!")

    logfilename = example.log_projerr(args.nreal, args.method, args.k).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("projerr", level=loglevel)

    # ### Coarse grid partition of omega
    coarse_grid_path = example.path_omega_coarse(args.k)
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": example.gdim})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)

    # ### Fine grid partition of omega
    path_omega = example.path_omega(args.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega.as_posix(), "r") as xdmf:
        omega_mesh = xdmf.read_mesh()
        omega_ct = xdmf.read_meshtags(omega_mesh, name="Cell tags")
        omega_ft = xdmf.read_meshtags(omega_mesh, name="mesh_tags")
    omega = RectangularDomain(omega_mesh, cell_tags=omega_ct, facet_tags=omega_ft)

    # ### Fine grid partition of omega_in
    path_omega_in = example.path_omega_in(args.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega_in.as_posix(), "r") as xdmf:
        omega_in_mesh = xdmf.read_mesh()
    omega_in = RectangularDomain(omega_in_mesh)

    logger.info(f"Discretizing transfer problem for k = {args.k:02} ...")
    osp_config = oversampling_config_factory(args.k)
    transfer, _ = discretize_transfer_problem(example, struct_grid, omega, omega_in, osp_config, debug=args.debug)


    # use same seeding as in actual training
    # ### Generate training seed for each of the 11 oversampling problems
    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    parameter_name = "R"

    myseeds_train = np.random.SeedSequence(example.training_set_seed).generate_state(11)
    ntrain = example.ntrain(args.k)
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=myseeds_train[args.k],
    )

    # seeds for the randomized range finder
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # ### Read basis and wrap as pymor object
    logger.info(f"Computing spectral basis with method {args.method} ...")
    epsilon_star = example.epsilon_star_projerr
    Nin = transfer.rhs.dofs.size
    basis = None
    svals = None

    if args.method == "hapod":
        from parageom.hapod import adaptive_rrf_normal

        snapshots = transfer.range.empty()
        spectral_basis_sizes = list()

        epsilon_alpha = np.sqrt(1 - example.omega**2) * epsilon_star
        epsilon_pod = np.sqrt(ntrain) * example.omega * epsilon_star

        # as number of testvectors we use Nin
        # the l2-mean error will be computed over set of testvectors

        for mu, seed_seq in zip(training_set, seed_seqs_rrf):
            with new_rng(seed_seq):
                transfer.assemble_operator(mu)
                rb = adaptive_rrf_normal(
                    logger,
                    transfer,
                    error_tol=example.rrf_ttol,
                    failure_tolerance=example.rrf_ftol,
                    num_testvecs=Nin,
                    l2_err=epsilon_alpha,
                )
                logger.info(f"\nSpectral Basis length: {len(rb)}.")
                spectral_basis_sizes.append(len(rb))
                snapshots.append(rb) # type: ignore
        logger.info(
            f"Average length of spectral basis: {np.average(spectral_basis_sizes)}."
        )
        logger.info("Computing final POD ...")
        basis, svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)  # type: ignore

    elif args.method == "heuristic":
        from parageom.heuristic import heuristic_range_finder

        # do the same for the testing set
        # only needed for the heuristic rrf
        myseeds_test = np.random.SeedSequence(example.testing_set_seed).generate_state(11)
        testing_set = sample_lhs(
            parameter_space,
            name=parameter_name,
            samples=ntrain, # same number of samples as in the training
            criterion="center",
            random_state=myseeds_test[args.k],
        )

        with new_rng(seed_seqs_rrf[0]):
            spectral_basis = heuristic_range_finder(
                logger,
                transfer,
                training_set,
                testing_set,
                error_tol=example.rrf_ttol,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=example.rrf_num_testvecs,
                l2_err=epsilon_star,
                compute_neumann=False,
                fext=None,
            )[0]
        basis = spectral_basis
    else:
        raise NotImplementedError

    basis_length = len(basis) # type: ignore
    orthonormal = np.allclose(
        basis.gramian(transfer.range_product), np.eye(basis_length), atol=1e-5
    )
    if not orthonormal:
        raise ValueError("Basis is not orthonormal wrt range product.")

    # Definition of (random) test set (μ) and test data (g)
    size_test_set = args.num_samples * args.num_testvecs
    logger.info(f"Computing test set of size {size_test_set}...")
    with new_rng(example.projerr_seed):
        test_set = parameter_space.sample_randomly(args.num_samples)
        test_data = transfer.range.empty(reserve=size_test_set)
        for mu in test_set:
            transfer.assemble_operator(mu)
            g = transfer.generate_random_boundary_data(args.num_testvecs, "normal")
            test_data.append(transfer.solve(g)) # type: ignore

    aerrs = defaultdict(list)
    rerrs = defaultdict(list)
    l2errs = defaultdict(list)

    def compute_norm(U, key, value):
        if key == "max":
            return U.amax()[1]
        else:
            assert key in (transfer.range_product.name, "euclidean") # type: ignore
            return U.norm(value)

    products = {transfer.range_product.name: transfer.range_product, "euclidean": None, "max": False} # type: ignore
    test_norms = {}
    for k, v in products.items():
        test_norms[k] = compute_norm(test_data, k, v)

    logger.info("Computing projection error ...")
    for N in range(basis_length + 1):
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
            l2_err = np.sum(error_norm**2.0) / size_test_set

            aerrs[k].append(np.max(error_norm))
            rerrs[k].append(np.max(rel_err))
            l2errs[k].append(l2_err)

    if args.show:
        import matplotlib.pyplot as plt

        plt.semilogy(np.arange(basis_length+1), l2errs[transfer.range_product.name], # type: ignore
                     label="l2-mean, "+transfer.range_product.name) # type: ignore
        plt.semilogy(np.arange(basis_length+1), rerrs[transfer.range_product.name], # type: ignore
                     label="rel. err, "+transfer.range_product.name) # type: ignore
        plt.legend()
        plt.show()

    if args.output is not None:
        np.savez(
            args.output,
            rerr_h1_semi=rerrs[transfer.range_product.name], # type: ignore
            rerr_euclidean=rerrs["euclidean"],
            rerr_max=rerrs["max"],
            aerr_h1_semi=aerrs[transfer.range_product.name], # type: ignore
            aerr_euclidean=aerrs["euclidean"],
            aerr_max=aerrs["max"],
            l2err_h1_semi=l2errs[transfer.range_product.name], # type: ignore
            l2err_euclidean=l2errs["euclidean"],
            l2err_max=l2errs["max"],
            svals=svals if svals is not None else np.array([], dtype=np.float32)
        )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser("Compute projection error over set of size `num_samples` * `num_testvecs`.")
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument("k", type=int, help="Use the k-th oversampling problem.")
    parser.add_argument("num_samples", type=int, help="Number of parameters used to define the test set.")
    parser.add_argument("num_testvecs", type=int, help="Number of test vectors used to define the test set.")
    parser.add_argument(
        "--output",
        type=str,
        help="Write absolute and relative projection error to file.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--show", action="store_true", help="Show projection error plot.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
