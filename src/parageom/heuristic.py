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
    logger.debug(f"{len(training_set)=}")
    while maxnorm > testlimit:
        basis_length = len(B)
        # randomly select mu from existing LHS design
        # mu_ind = rng.integers(0, len(training_set))
        # FIXME
        # figure out what is the best stratgey for drawing μ from LHS design
        # logger.debug(f"{mu_ind=}")
        # mu = training_set.pop(mu_ind)
        mu = training_set.pop(0)
        training_samples.append(mu)
        tp.assemble_operator(mu)
        # FIXME
        # instead of adjusting the parameter samples in the LHS design
        # it is simpler to simply use more boundary data for the same sample
        v = tp.generate_random_boundary_data(3, distribution, options=sampling_options)

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

    # FIXME
    # I am not sure if I should simply sample mu randomly inside the while loop
    # `samples` will have an influence on the LHS design
    # but still this should be better than walk through parameter space randomly?
    # the downside is that I cannot guarantee that the training set will be larger
    # than the number of iterations required
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=training_seeds[args.configuration],
    )
    # training_set.extend(
    #         sample_lhs(parameter_space, name=parameter_name, samples=ntrain, criterion="center", random_state=training_seeds[args.configuration])
    #         )
    # breakpoint()
    # assert len(training_set) == 2 * ntrain

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
    training_set_length = len(training_set)
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
    assert len(training_set) + len(training_samples) == training_set_length

    # ### Compute Neumann Modes
    neumann_snapshots = spectral_basis.space.empty(reserve=len(training_samples))
    for mu in training_samples:
        transfer.assemble_operator(mu)
        U_neumann = transfer.op.apply_inverse(FEXT)
        # u_vec = transfer._u.x.petsc_vec  # type: ignore
        # u_vec.array[:] = U_neumann.to_numpy().flatten()
        # transfer._u.x.scatter_forward()  # type: ignore

        # ### restrict full solution to target subdomain
        # transfer._u_in.interpolate(
        #     transfer._u, nmm_interpolation_data=transfer._interp_data
        # )  # type: ignore
        # transfer._u_in.x.scatter_forward()  # type: ignore
        # U_in_neumann = transfer.range.make_array([transfer._u_in.x.petsc_vec.copy()])  # type: ignore
        U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))

        # ### Remove kernel after restriction to target subdomain
        U_orth = orthogonal_part(
            U_in_neumann, transfer.kernel, product=transfer.range_product, orthonormal=True
        )
        neumann_snapshots.append(U_orth)

    assert np.allclose(
        spectral_basis.gramian(transfer.range_product), np.eye(len(spectral_basis))
    )
    logger.info("Extending spectral basis by Neumann snapshots via GS ...")
    basis_length = len(spectral_basis)
    n_neumann = len(neumann_snapshots)
    # extend_basis(neumann_snapshots, spectral_basis, product=transfer.range_product, method="gram_schmidt")
    # gram_schmidt(neumann_snapshots, transfer.range_product, atol=0, rtol=0, copy=False)
    N_proj_err = neumann_snapshots - spectral_basis.lincomb(
        neumann_snapshots.inner(spectral_basis, transfer.range_product)
    )
    neumann_modes, _ = pod(
        N_proj_err,
        modes=len(neumann_snapshots),
        product=transfer.range_product,
        rtol=example.pod_rtol,
        orth_tol=np.inf,
    )

    spectral_basis.append(neumann_modes)
    gram_schmidt(
        spectral_basis,
        product=transfer.range_product,
        offset=basis_length,
        check=False,
        copy=False,
    )
    assert np.allclose(spectral_basis.gramian(transfer.range_product),
                       np.eye(len(spectral_basis)))

    logger.info(f"Spectral basis size: {basis_length}.")
    # logger.info(f"Neumann modes: {n_neumann}.")
    logger.info(f"Neumann modes: {len(neumann_modes)}/{n_neumann}.")
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
