from mpi4py import MPI
from multi.domain import StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import orthogonal_part
import numpy as np
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.parameters.base import ParameterSpace
from pymor.tools.random import new_rng
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv


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

    while (maxnorm > testlimit) and (l2 > l2_err**2.0):
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
    logger.info(f"{maxnorm < testlimit =}\t{maxnorm=}\t{testlimit=}")
    logger.info(f"{l2 < l2_err ** 2 =}\t{l2=}")
    logger.info(f"Finished RRF in {len(B)} iterations ({reason=}).")

    return B


def main(args):
    from parageom.tasks import example
    from parageom.lhs import sample_lhs
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    logfilename = example.log_basis_construction(args.nreal, "hapod", args.k).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("hapod", level=loglevel)

    # ### Coarse grid partition
    coarse_grid_path = example.coarse_grid("global")
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": example.gdim})[0]
    struct_grid_gl = StructuredQuadGrid(coarse_domain)

    logger.info(f"Discretizing transfer problem for k = {args.k:02} ...")
    osp_config = oversampling_config_factory(args.k)
    transfer, fext = discretize_transfer_problem(example, struct_grid_gl, osp_config, debug=args.debug)

    # ### Generate training seed for each of the 11 oversampling problems
    myseeds = np.random.SeedSequence(example.training_set_seed).generate_state(11)

    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    parameter_name = "R"
    ntrain = example.ntrain(args.k)
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=myseeds[args.k],
    )
    logger.info("Starting range approximation of transfer operators" f" for training set of size {len(training_set)}.")

    # ### Generate random seed for each specific mu in the training set
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # start range approximation with `transfer` and `F_ext`
    require_neumann_data = np.any(np.nonzero(fext.to_numpy())[1])
    if 0 in osp_config.cells_omega:
        assert require_neumann_data
    else:
        assert not require_neumann_data

    assert len(training_set) == len(seed_seqs_rrf)
    snapshots = transfer.range.empty()
    neumann_snapshots = transfer.range.empty(reserve=len(training_set))
    spectral_basis_sizes = list()

    epsilon_star = example.epsilon_star["hapod"]
    Nin = transfer.rhs.dofs.size
    epsilon_alpha = np.sqrt(1 - example.omega**2.0) * epsilon_star
    epsilon_pod = np.sqrt(ntrain) * example.omega * epsilon_star

    for mu, seed_seq in zip(training_set, seed_seqs_rrf):
        with new_rng(seed_seq):
            transfer.assemble_operator(mu)
            basis = adaptive_rrf_normal(
                logger,
                transfer,
                error_tol=example.rrf_ttol / example.l_char,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=Nin,
                l2_err=epsilon_alpha,
            )
            logger.info(f"\nSpectral Basis length: {len(basis)}.")
            spectral_basis_sizes.append(len(basis))
            snapshots.append(basis)  # type: ignore

            if require_neumann_data:
                logger.info("\nSolving for additional Neumann mode ...")
                U_neumann = transfer.op.apply_inverse(fext)
                U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction)) # type: ignore

                # ### Remove kernel after restriction to target subdomain
                if transfer.kernel is not None:
                    U_orth = orthogonal_part(
                        U_in_neumann,
                        transfer.kernel,
                        product=None,
                        orthonormal=True,
                    )
                else:
                    U_orth = U_in_neumann
                neumann_snapshots.append(U_orth)  # type: ignore

    logger.info(f"Average length of spectral basis: {np.average(spectral_basis_sizes)}.")
    if len(neumann_snapshots) > 0: # type: ignore
        logger.info("Appending Neumann snapshots to global snapshot set.")
        snapshots.append(neumann_snapshots) # type: ignore

    logger.info("Computing final POD")
    spectral_modes, spectral_svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod) # type: ignore

    if logger.level == 10:  # DEBUG
        from pymor.bindings.fenicsx import FenicsxVisualizer

        viz = FenicsxVisualizer(transfer.range)
        hapod_modes_xdmf = example.hapod_modes_xdmf(args.nreal, args.k).as_posix()
        viz.visualize(spectral_modes, filename=hapod_modes_xdmf)

    np.save(
        example.hapod_modes_npy(args.nreal, args.k),
        spectral_modes.to_numpy(),
    )
    np.save(
        example.hapod_singular_values(args.nreal, args.k),
        spectral_svals,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Oversampling for ParaGeom example using HAPOD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nreal", type=int, help="The n-th realization of the problem.")
    parser.add_argument("k", type=int, help="The oversampling problem for target subdomain Ω_in^k.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
