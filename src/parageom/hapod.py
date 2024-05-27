import os
from typing import Optional

# from itertools import repeat
from pathlib import Path

# import concurrent.futures
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv
from time import perf_counter

from mpi4py import MPI
import dolfinx as df

from pymor.bindings.fenicsx import (
    FenicsxVectorSpace,
    FenicsxMatrixOperator,
    FenicsxVisualizer,
)
from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import NumpyConversionOperator
from pymor.tools.random import new_rng

from multi.io import read_mesh
from multi.problems import TransferProblem
from multi.projection import orthogonal_part
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem


def adaptive_rrf_normal(
    logger,
    transfer_problem: TransferProblem,
    source_product: Optional[Operator] = None,
    range_product: Optional[Operator] = None,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    **sampling_options,
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

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution)
    M = tp.solve(R)
    B = tp.range.empty()
    maxnorm = np.inf
    while maxnorm > testlimit:
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, distribution, **sampling_options)

        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        logger.debug(f"{maxnorm=}")

    return B


def approximate_range(args, example, transfer_problem, logfilename):
    """Approximates range of the transfer operator.

    Args:
        args: Program args.
        example: Example data class.
        transfer_problem: The transfer problem whose's operator range is approximated.
        logfilename: Logfile.

    """
    from .definitions import BeamProblem

    distribution = args.distribution
    configuration = args.configuration

    logger = getLogger("range_approximation", level="INFO", filename=logfilename)

    if distribution != "normal":
        raise NotImplementedError

    # Multiscale problem definition
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    cell_index = beam_problem.config_to_cell(configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    dirichlet = beam_problem.get_dirichlet(cell_index)

    ttol = example.rrf_ttol
    ftol = example.rrf_ftol
    num_testvecs = example.rrf_num_testvecs
    source_product = transfer_problem.source_product
    range_product = transfer_problem.range_product

    # TODO approximate range in context of new_rng (if number of realizations > 1)
    basis = adaptive_rrf_normal(
        logger,
        transfer_problem,
        source_product=source_product,
        range_product=range_product,
        error_tol=ttol,
        failure_tolerance=ftol,
        num_testvecs=num_testvecs,
    )
    logger.info(f"\nSpectral Basis length: {len(basis)}.")

    # ### Add Solution of Neumann Problem
    # Neumann problem
    neumann_problem = transfer_problem.problem
    neumann_problem.clear_bcs()
    omega = neumann_problem.domain

    # Add Neumann bc
    traction_y = example.traction_y
    loading = df.fem.Constant(
        omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(-traction_y))
    )
    top_facets = int(14)  # see locmor.py l. 95
    neumann_problem.add_neumann_bc(top_facets, loading)

    # Add zero boundary conditions on gamma out
    zero = df.fem.Constant(
        omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(0.0))
    )
    neumann_problem.add_dirichlet_bc(zero, gamma_out, method="geometrical")

    # Add homogeneous Dirichlet BCs if present
    if dirichlet is not None:
        neumann_problem.add_dirichlet_bc(**dirichlet)

    # ### Solve
    neumann_problem.setup_solver()
    u_neumann = neumann_problem.solve()

    # ### Restrict to target subdomain
    u_in = df.fem.Function(transfer_problem.range.V)
    u_in.interpolate(
        u_neumann,
        nmm_interpolation_data=df.fem.create_nonmatching_meshes_interpolation_data(
            u_in.function_space.mesh,
            u_in.function_space.element,
            u_neumann.function_space.mesh,
        ),
    )

    U_in = transfer_problem.range.make_array([u_in.vector])
    if transfer_problem.kernel is None:
        U_orth = U_in
    else:
        U_orth = orthogonal_part(
            U_in,
            transfer_problem.kernel,
            product=transfer_problem.range_product,
            orthonormal=True,
        )

    logger.info("Extension of basis by Neumann Mode ...")
    basis_length = len(basis)
    basis.append(U_orth)
    gram_schmidt(
        basis,
        transfer_problem.range_product,
        atol=0,
        rtol=0,
        offset=basis_length,
        copy=False,
    )

    # ---- ensemble mean subtraction ----
    # num_snapshots = len(fullsnapshots)
    # snaps = fullsnapshots.copy()
    # snaps.scal(1.0 / num_snapshots)
    # ensemble_mean = sum(snaps)
    # fullsnapshots.axpy(-1.0, ensemble_mean)

    # ### Conversion to picklable objects
    # passing data along processes requires picklable data
    # fenics stuff is not picklable ...
    # cop = NumpyConversionOperator(basis.space, direction="to_numpy") # type: ignore
    # basis_numpy = cop.apply(basis)
    # product_matrix = csr_array(transfer_problem.range_product.matrix.getValuesCSR()[::-1]) # type: ignore
    # product_numpy = NumpyMatrixOperator(product_matrix, source_id=basis_numpy.space.id,
    #                                     range_id=basis_numpy.space.id)
    #

    # maybe go back to conversion if this is actually run in parallel
    return basis


def main(args):
    from .tasks import example
    from .locmor import discretize_oversampling_problem
    from .auxiliary_problem import GlobalAuxiliaryProblem

    method = Path(__file__).stem  # hapod
    logfilename = example.log_edge_basis(
        args.nreal, method, args.distribution, args.configuration
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level="INFO")

    ntrain = example.ntrain(args.configuration)
    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {ntrain}."
    )

    # FIXME
    # remove task_training_set
    # generate training set here instead of dodo.py

    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=args.max_workers
    # ) as executor:
    #     results = executor.map(
    #         spawn_rng(approximate_range),
    #         repeat(beam),
    #         training_set,
    #         repeat(args.configuration),
    #         repeat(args.distribution),
    #     )

    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # ### Oversampling Domain
    domain, ct, ft = read_mesh(example.parent_domain(args.configuration), MPI.COMM_SELF, gdim=example.gdim)
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)
    omega.create_facet_tags(
        {"bottom": int(11), "left": int(12), "right": int(13), "top": int(14)}
    )

    aux_tags = None
    if args.configuration == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]

    elif args.configuration == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        aux_tags = [15, 16]

    elif args.configuration == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        aux_tags = [15, 16]
    else:
        raise NotImplementedError

    # ### Structured coarse grid
    grid, _, _ = read_mesh(example.coarse_grid(args.configuration), MPI.COMM_SELF, gdim=example.gdim)
    coarse_grid = StructuredQuadGrid(grid)

    # ### Auxiliary Problem
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu)
    V = df.fem.functionspace(omega.grid, ("P", example.geom_deg, (example.gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=mat)
    auxiliary_problem = GlobalAuxiliaryProblem(problem, aux_tags, example.parameters[args.configuration], coarse_grid)
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Transfer Problem

    # definition of FenicsxMatrixBasedOperator
    # definition of rhs (Dirichlet Lift Operator?)
    # range space (requires (correctly translated) target subdomain)
    # gamma out?
    # source product
    # range product
    # kernel

    # no need to setup a LinElaSubProblem, because coarse/fine scale split is not used

    # first sample
    index = 0
    tic = perf_counter()
    transfer_problem = discretize_oversampling_problem(
        example, args.configuration, index
    )
    # FIXME
    # this might be problematic as in theory each range product
    # is assembled on a different mesh (target subdomain)
    range_product = transfer_problem.range_product
    pid = os.getpid()
    logger.debug(f"{pid=},\tDiscretized transfer problem in {perf_counter()-tic}.")

    logger.info(
        f"{pid=},\tApproximating range of T for {index=} using {args.distribution=}.\n"
    )
    ss = seed_seqs_rrf[index]
    with new_rng(ss):
        basis = approximate_range(args, example, transfer_problem, logfilename)

    snapshots = basis.space.empty()
    snapshots.append(basis)

    for index in range(1, ntrain):
        transfer_problem = discretize_oversampling_problem(
            example, args.configuration, index
        )
        ss = seed_seqs_rrf[index]
        with new_rng(ss):
            other = approximate_range(args, example, transfer_problem, logfilename)
        # other not in snapshots.space
        B = snapshots.space.from_numpy(other.to_numpy())
        snapshots.append(B)

    pod_modes, pod_svals = pod(snapshots, product=range_product, rtol=example.pod_rtol)  # type: ignore

    viz = FenicsxVisualizer(pod_modes.space)
    viz.visualize(
        pod_modes,
        filename=example.hapod_modes_xdmf(
            args.nreal, args.distribution, args.configuration
        ),
    )
    np.save(
        example.hapod_modes_npy(args.nreal, args.distribution, args.configuration),
        pod_modes.to_numpy(),
    )
    np.save(
        example.hapod_singular_values(
            args.nreal, args.distribution, args.configuration
        ),
        pod_svals,
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
    parser.add_argument(
        "--max_workers", type=int, default=4, help="The max number of workers."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
