from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.bindings.fenicsx import (
    FenicsxVisualizer
)
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.tools.random import new_rng
from pymor.parameters.base import ParameterSpace

from multi.projection import orthogonal_part


def heuristic_range_finder(
    logger,
    transfer_problem,
    training_set,
    testing_set,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    sampling_options=None,
):
    """Heuristic range approximation."""

    tp = transfer_problem
    distribution = "normal"
    sampling_options = sampling_options or {}

    source_product = tp.source_product
    range_product = tp.range_product
    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)

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
            (source_product.source.dim, source_product.range.dim),  # type: ignore
            matvec=mv,  # type: ignore
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim),  # type: ignore
            matvec=mvinv,  # type: ignore
        )
        lambda_min = eigsh(
            L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
        )[0]

    # ### Compute non-parametric testlimit
    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = tp.rhs.dofs.size
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )
    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    logger.debug(f"Computing test set of size {len(testing_set) * num_testvecs}.")
    M = tp.range.empty()  # global test set
    for mu in testing_set:
        tp.assemble_operator(mu)
        R = tp.generate_random_boundary_data(
            count=num_testvecs, distribution=distribution, options=sampling_options
        )
        M.append(tp.solve(R))

    # rng = get_rng()  # current RNG
    training_samples = []  # parameter values used in the training
    B = tp.range.empty()
    maxnorm = np.inf
    num_iter = 0
    ntrain = len(training_set)
    # re-use {mu_0, ..., mu_ntrain} until target tolerance is reached
    mu_j = np.hstack((np.arange(ntrain, dtype=np.int32),) * 3)
    logger.debug(f"{ntrain=}")
    while maxnorm > testlimit:
        basis_length = len(B)
        j = mu_j[num_iter]
        mu = training_set[j]
        # randomly select mu from existing LHS design
        # mu_ind = rng.integers(0, len(training_set))
        # FIXME
        # figure out what is the best stratgey for drawing μ from LHS design
        # logger.debug(f"{mu_ind=}")
        # mu = training_set.pop(mu_ind)
        # mu = training_set.pop(0)
        training_samples.append(mu)
        tp.assemble_operator(mu)
        # FIXME
        # instead of adjusting the parameter samples in the LHS design
        # it is simpler to simply use more boundary data for the same sample
        v = tp.generate_random_boundary_data(1, distribution, options=sampling_options)

        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        num_iter += 1
        logger.debug(f"{num_iter=}\t{maxnorm=}")
    logger.info(f"Finished heuristic range approx. in {num_iter} iterations.")

    return B, training_samples


def main(args):
    from .tasks import example
    from .lhs import sample_lhs
    from .locmor import discretize_transfer_problem

    method = Path(__file__).stem  # heuristic
    logfilename = example.log_basis_construction(
        args.nreal, method, args.distribution, args.configuration
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level="DEBUG")

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
        example.parameters[args.configuration], example.mu_range
    )
    parameter_name = list(example.parameters[args.configuration].keys())[0]
    ntrain = example.ntrain(args.configuration)

    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=training_seeds[args.configuration],
    )

    # NOTE
    # The testing set for the heuristic range approximation should have
    # the same size as the training set in the HAPOD, such that the
    # range of the same number of different transfer operators is approximated
    # in both variants.
    testing_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=testing_seeds[args.configuration],
    )
    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
    )

    # ### Generate random seed to draw random samples in the range finder algorithm
    # in the case of the heuristic range finder there is only one while loop
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(1)

    transfer, FEXT = discretize_transfer_problem(example, args.configuration)

    # ### Heuristic range approximation
    logger.debug(f"{seed_seqs_rrf[0]=}")
    with new_rng(seed_seqs_rrf[0]):
        spectral_basis, training_samples = heuristic_range_finder(
            logger,
            transfer,
            training_set,
            testing_set,
            error_tol=example.rrf_ttol,
            failure_tolerance=example.rrf_ftol,
            num_testvecs=example.rrf_num_testvecs,
            sampling_options={"scale":0.1},
        )

    # ### Compute Neumann Modes
    # restrict to ntrain, otherwise would just compute the same data twice
    neumann_snapshots = spectral_basis.space.empty(reserve=len(training_samples[:ntrain]))
    for mu in training_samples[:ntrain]:
        transfer.assemble_operator(mu)
        U_neumann = transfer.op.apply_inverse(FEXT)
        U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))

        # ### Remove kernel after restriction to target subdomain
        U_orth = orthogonal_part(
            U_in_neumann, transfer.kernel, product=None, orthonormal=True
        )
        neumann_snapshots.append(U_orth)

    logger.info("Extending spectral basis by Neumann snapshots via GS ...")
    basis_length = len(spectral_basis)
    spectral_basis.append(neumann_snapshots)
    gram_schmidt(
        spectral_basis,
        product=transfer.range_product,
        offset=basis_length,
        check=False,
        copy=False,
    )

    logger.info(f"Spectral basis size: {basis_length}.")
    logger.info(f"Neumann snapshots: {len(neumann_snapshots)}")
    logger.info(f"Final basis length: {len(spectral_basis)}.")

    viz = FenicsxVisualizer(spectral_basis.space)
    viz.visualize(
        spectral_basis,
        filename=example.heuristic_modes_xdmf(
            args.nreal, args.distribution, args.configuration
        ),
    )
    np.save(
        example.heuristic_modes_npy(args.nreal, args.distribution, args.configuration),
        spectral_basis.to_numpy(),
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Computes fine scale edge basis functions via transfer problems and subsequently the POD of these sets of basis functions.",
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
    parser.add_argument("nreal", type=int, help="The n-th realization of the problem.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
