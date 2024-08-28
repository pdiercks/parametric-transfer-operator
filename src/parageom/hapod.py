import sys

from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.tools.random import new_rng
from pymor.parameters.base import ParameterSpace

from multi.projection import orthogonal_part


def adaptive_rrf_normal(
    logger,
    transfer_problem,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    l2_err: float = 0.0,
    sampling_options=None,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is the
    |VectorArray| `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the
    norm denotes the operator norm. The inner product of the range of
    `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    NOTE
    ----
    Instead of a transfer operator A, a transfer problem is used.
    (see multi.problem.TransferProblem)
    The image Av = A.apply(v) is equivalent to the restriction
    of the full solution to the target domain Ω_in, i.e.
        U = transfer_problem.solve(v)

    Parameters
    ----------
    transfer_problem
        The transfer problem associated with a (transfer) |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    error_tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    sampling_options
        Optional keyword arguments for the generation of
        random samples (training data).
        see `_create_random_values`.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.

    """

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
        lambda_min = eigsh(L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv)[0]

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = tp.rhs.dofs.size
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    def l2_mean(U, basis, product=None):
        error = U - basis.lincomb(basis.inner(U, product).T)
        norm = error.norm2(product)
        return np.sum(norm) / len(U)

    R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution, options=sampling_options)
    M = tp.solve(R)
    B = tp.range.empty()
    maxnorm = np.inf
    l2 = np.sum(M.norm2(range_product)) / len(M)

    l2_errors = [l2,]
    max_norms = [maxnorm,]

    while (l2 > l2_err**2.0):
    # while (maxnorm > testlimit) and (l2 > l2_err**2.0):
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, distribution, options=sampling_options)

        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        l2 = l2_mean(M, B, range_product)
        logger.debug(f"{maxnorm=}")

        l2_errors.append(l2)
        max_norms.append(maxnorm)

    reason = "maxnorm" if maxnorm < testlimit else "l2err"
    logger.info(f"{maxnorm < testlimit =}\t{maxnorm=}")
    logger.info(f"{l2 < l2_err ** 2 =}\t{l2=}")
    logger.info(f"Finished RRF in {len(B)} iterations ({reason=}).")

    return B


def main(args):
    from .tasks import example
    from .lhs import sample_lhs
    from .locmor import discretize_transfer_problem

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    method = Path(__file__).stem  # hapod
    logfilename = example.log_basis_construction(args.nreal, method, args.distribution, args.configuration).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level=loglevel)

    # ### Generate training seed for each configuration
    training_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.training_set_seed).generate_state(len(example.configurations)),
    ):
        training_seeds[cfg] = rndseed

    parameter_space = ParameterSpace(example.parameters[args.configuration], example.mu_range)
    parameter_name = list(example.parameters[args.configuration].keys())[0]
    ntrain = example.ntrain(args.configuration)
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=training_seeds[args.configuration],
    )
    logger.info("Starting range approximation of transfer operators" f" for training set of size {len(training_set)}.")

    # ### Generate random seed for each specific mu in the training set
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    transfer, FEXT = discretize_transfer_problem(example, args.configuration)
    require_neumann_data = np.any(np.nonzero(FEXT.to_numpy())[1])
    if args.configuration == "left":
        assert require_neumann_data
    else:
        assert not require_neumann_data

    assert len(training_set) == len(seed_seqs_rrf)
    snapshots = transfer.range.empty()
    neumann_snapshots = transfer.range.empty(reserve=len(training_set))
    spectral_basis_sizes = list()

    epsilon_star = example.epsilon_star["hapod"]
    Nin = transfer.rhs.dofs.size
    epsilon_alpha = np.sqrt(Nin) * np.sqrt(1 - example.omega**2.0) * epsilon_star
    epsilon_pod = epsilon_star * np.sqrt(Nin * ntrain)

    # scaling
    epsilon_alpha /= example.l_char
    epsilon_pod /= example.l_char

    for mu, seed_seq in zip(training_set, seed_seqs_rrf):
        with new_rng(seed_seq):
            transfer.assemble_operator(mu)
            basis = adaptive_rrf_normal(
                logger,
                transfer,
                error_tol=example.rrf_ttol / example.l_char,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=example.rrf_num_testvecs,
                l2_err=epsilon_alpha,
                sampling_options={"scale": 0.1},
            )
            logger.info(f"\nSpectral Basis length: {len(basis)}.")
            spectral_basis_sizes.append(len(basis))
            snapshots.append(basis)  # type: ignore

            if require_neumann_data:
                logger.info("\nSolving for additional Neumann mode ...")
                U_neumann = transfer.op.apply_inverse(FEXT)
                U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))

                # ### Remove kernel after restriction to target subdomain
                U_orth = orthogonal_part(
                    U_in_neumann,
                    transfer.kernel,
                    product=None,
                    orthonormal=True,
                )
                neumann_snapshots.append(U_orth)  # type: ignore

    logger.info(f"Average length of spectral basis: {np.average(spectral_basis_sizes)}.")
    with logger.block("Computing POD of spectral bases ..."):
        spectral_modes, spectral_svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)

    basis_length = len(spectral_modes)
    if require_neumann_data:
        with logger.block("Computing POD of neumann snapshots ..."):
            neumann_modes, neumann_svals = pod(
                neumann_snapshots, product=transfer.range_product, rtol=example.neumann_rtol
            )

        with logger.block("Extending spectral basis by Neumann modes via GS ..."):
            spectral_modes.append(neumann_modes)
            gram_schmidt(spectral_modes, product=transfer.range_product, offset=basis_length, check=False, copy=False)
    else:
        neumann_modes = []
        neumann_snapshots = []
        neumann_svals = []

    logger.info(f"Spectral basis size (after POD): {basis_length}.")
    logger.info(f"Neumann modes/snapshots: {len(neumann_modes)}/{len(neumann_snapshots)}")
    logger.info(f"Final basis length: {len(spectral_modes)}.")

    if logger.level == 10:  # DEBUG
        from pymor.bindings.fenicsx import FenicsxVisualizer

        viz = FenicsxVisualizer(transfer.range)
        viz.visualize(spectral_modes, filename=f"hapod_{args.configuration}.xdmf")

    np.save(
        example.hapod_modes_npy(args.nreal, args.distribution, args.configuration),
        spectral_modes.to_numpy(),
    )
    np.save(
        example.hapod_singular_values(args.nreal, args.distribution, args.configuration),
        spectral_svals,
    )
    if np.any(neumann_svals) and args.configuration == "left":
        np.save(example.hapod_neumann_svals(args.nreal, args.distribution, args.configuration), neumann_svals)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Computes fine scale edge basis functions via transfer problems and subsequently the POD of these sets of basis functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("distribution", type=str, help="The distribution to draw samples from.")
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    parser.add_argument("nreal", type=int, help="The n-th realization of the problem.")
    parser.add_argument("--max_workers", type=int, default=4, help="The max number of workers.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
