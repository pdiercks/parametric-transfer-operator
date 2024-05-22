import os
import json
from typing import Optional

# from itertools import repeat
from pathlib import Path
from collections import defaultdict

# import concurrent.futures
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv
from time import perf_counter

from dolfinx import mesh, fem, default_scalar_type

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import NumpyConversionOperator
from pymor.tools.random import new_rng
from pymor.reductors.basic import extend_basis
from pymor.tools.table import format_table

from multi.problems import TransferProblem
from multi.projection import orthogonal_part, fine_scale_part
from multi.dofmap import QuadrilateralDofLayout
from multi.product import InnerProduct
from multi.shapes import NumpyLine


def adaptive_edge_rrf_normal(
    logger,
    transfer_problem: TransferProblem,
    active_edges,
    source_product: Optional[Operator] = None,
    range_product: Optional[str] = None,
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
    active_edges
        A list of edges of the target subdomain.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        A str specifying which inner product to use.
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
    pod_bases
        A dict which contains a |VectorArray| for each 'active edge'.
        The |VectorArray| contains the POD basis which
        span approximates the image of the transfer operator A
        restricted to the respective edge.
    range_products
        The inner product operators constructed in the edge
        range spaces.

    """

    tp = transfer_problem

    distribution = "normal"

    assert source_product is None or isinstance(source_product, Operator)
    range_product = range_product or "h1"

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

    # ### test set
    R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution)
    M = tp.solve(R)

    dof_layout = QuadrilateralDofLayout()
    edge_index_map = dof_layout.local_edge_index_map

    # ### initialize data structures
    test_set = {}
    range_spaces = {}
    range_products = {}
    pod_bases = {}
    maxnorm = np.array([], dtype=float)
    edges = np.array([], dtype=str)
    coarse_basis = {}
    # the dofs for vertices on the boundary of the edge
    edge_boundary_dofs = {}

    start = perf_counter()
    for i in range(dof_layout.num_entities[1]):
        edge = edge_index_map[i]
        edges = np.append(edges, edge)

        edge_mesh = tp.subproblem.domain.fine_edge_grid[edge]
        edge_space = tp.subproblem.edge_spaces["fine"][edge]
        range_spaces[edge] = FenicsxVectorSpace(edge_space)

        # ### create dirichletbc for range product
        facet_dim = edge_mesh.topology.dim - 1
        vertices = mesh.locate_entities_boundary(
            edge_mesh, facet_dim, lambda x: np.full(x[0].shape, True, dtype=bool)
        )
        _dofs_ = fem.locate_dofs_topological(edge_space, facet_dim, vertices)
        gdim = tp.subproblem.domain.grid.geometry.dim
        range_bc = fem.dirichletbc(
            np.array((0,) * gdim, dtype=default_scalar_type), _dofs_, edge_space
        )
        edge_boundary_dofs[edge] = range_bc._cpp_object.dof_indices()[0]

        # ### range product
        inner_product = InnerProduct(edge_space, range_product, bcs=(range_bc,))
        range_product_op = FenicsxMatrixOperator(
            inner_product.assemble_matrix(), edge_space, edge_space
        )
        range_products[edge] = range_product_op

        # ### compute coarse scale edge basis
        nodes = mesh.compute_midpoints(edge_mesh, facet_dim, vertices)
        nodes = np.around(nodes, decimals=3)

        component = 0
        if edge in ("left", "right"):
            component = 1

        line_element = NumpyLine(nodes[:, component])
        shape_funcs = line_element.interpolate(edge_space, component)
        N = range_spaces[edge].from_numpy(shape_funcs)
        coarse_basis[edge] = N

        # ### edge test sets
        dofs = tp.subproblem.V_to_L[edge]
        test_set[edge] = range_spaces[edge].from_numpy(M.dofs(dofs))
        # subtract coarse scale part
        test_cvals = test_set[edge].dofs(edge_boundary_dofs[edge])
        test_set[edge] -= N.lincomb(test_cvals)
        assert np.isclose(np.sum(test_set[edge].dofs(edge_boundary_dofs[edge])), 1e-9)

        # ### initialize maxnorm
        if edge in active_edges:
            maxnorm = np.append(maxnorm, np.inf)
            # ### pod bases
            pod_bases[edge] = range_spaces[edge].empty()
        else:
            maxnorm = np.append(maxnorm, 0.0)

    end = perf_counter()
    logger.debug(f"Preparing stuff took t={end-start}s.")

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = len(tp._bc_dofs_gamma_out)
    testfail = np.array(
        [
            failure_tolerance / min(num_source_dofs, space.dim)
            for space in range_spaces.values()
        ]
    )
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    snapshots = M.space.empty()

    num_solves = 0
    while np.any(maxnorm > testlimit):
        v = tp.generate_random_boundary_data(1, distribution, **sampling_options)

        U = tp.solve(v)
        snapshots.append(U)
        num_solves += 1

        target_edges = edges[maxnorm > testlimit]
        for edge in target_edges:
            B = pod_bases[edge]
            edge_space = range_spaces[edge]
            # restrict the training sample to the edge
            Udofs = edge_space.from_numpy(U.dofs(tp.subproblem.V_to_L[edge]))
            coarse_values = Udofs.dofs(edge_boundary_dofs[edge])
            U_fine = Udofs - coarse_basis[edge].lincomb(coarse_values)

            # extend pod basis
            extend_basis(U_fine, B, product=range_products[edge], method="gram_schmidt")

            # orthonormalize test set wrt pod basis
            M = test_set[edge]
            M -= B.lincomb(B.inner(M, range_products[edge]).T)

            norm = M.norm(range_products[edge])
            maxnorm[edge_index_map[edge]] = np.max(norm)

        logger.debug(f"{maxnorm=}")

    return pod_bases, range_products, num_solves, snapshots


def approximate_range(args, example, index, logfilename):
    """Approximates range of the transfer operator.

    Args:
        args: Program args.
        example: Example data class.
        index: Index of training sample.
        logfilename: Logfile.

    """
    from .definitions import BeamProblem
    from .locmor import discretize_oversampling_problem

    distribution = args.distribution
    configuration = args.configuration

    logger = getLogger("range_approximation", level="INFO", filename=logfilename)
    pid = os.getpid()
    logger.info(
        f"{pid=},\tApproximating range of T for {index=} using {distribution=}.\n"
    )

    tic = perf_counter()
    transfer_problem = discretize_oversampling_problem(example, configuration, index)
    logger.info(f"{pid=},\tDiscretized transfer problem in {perf_counter()-tic}.")

    if distribution != "normal":
        raise NotImplementedError

    # Multiscale problem definition
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    cell_index = beam_problem.config_to_cell(configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    dirichlet = beam_problem.get_dirichlet(cell_index)
    # determine basis for each edge
    # in contrast to beam_problem.active_edges(cell_index)
    active_edges = set(["bottom", "left", "right", "top"])

    ttol = example.rrf_ttol
    ftol = example.rrf_ftol
    num_testvecs = example.rrf_num_testvecs
    source_product = transfer_problem.source_product
    range_product = example.range_product

    # TODO approximate range in context of new_rng (if number of realizations > 1)
    edge_bases, range_products, num_solves, fullsnapshots = adaptive_edge_rrf_normal(
        logger,
        transfer_problem,
        active_edges,
        source_product=source_product,
        range_product=range_product,
        error_tol=ttol,
        failure_tolerance=ftol,
        num_testvecs=num_testvecs,
    )
    logger.info(f"\nNumber of transfer operator evaluations: {num_solves}.")

    table_basis_length = []
    table_basis_length.append(["Edge", "Basis length"])
    for edge, rb in edge_bases.items():
        table_basis_length.append([edge, len(rb)])
    table_title = f"\nNumber of basis functions after rrf ({pid=})."
    logger.info(format_table(table_basis_length, title=table_title))

    # ### Add Solution of Neumann Problem
    # Neumann problem
    neumann_problem = transfer_problem.problem
    neumann_problem.clear_bcs()
    omega = neumann_problem.domain

    # Add Neumann bc
    traction_y = example.traction_y
    loading = fem.Constant(
        omega.grid, (default_scalar_type(0.0), default_scalar_type(-traction_y))
    )
    top_facets = int(14)  # see locmor.py l. 95
    neumann_problem.add_neumann_bc(top_facets, loading)

    # Add zero boundary conditions on gamma out
    zero = fem.Constant(
        omega.grid, (default_scalar_type(0.0), default_scalar_type(0.0))
    )
    neumann_problem.add_dirichlet_bc(zero, gamma_out, method="geometrical")

    # Add homogeneous Dirichlet BCs if present
    if dirichlet is not None:
        neumann_problem.add_dirichlet_bc(**dirichlet)

    # ### Solve
    neumann_problem.setup_solver()
    u_neumann = neumann_problem.solve()

    # ### Restrict to target subdomain
    u_in = fem.Function(transfer_problem.range.V)
    u_in.interpolate(
        u_neumann,
        nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
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

    fullsnapshots.append(U_orth)

    # ---- ensemble mean subtraction ----
    num_snapshots = len(fullsnapshots)
    snaps = fullsnapshots.copy()
    snaps.scal(1. / num_snapshots)
    ensemble_mean = sum(snaps)
    fullsnapshots.axpy(-1., ensemble_mean)

    viz = FenicsxVisualizer(fullsnapshots.space)
    viz.visualize(fullsnapshots, filename=example.hapod_snapshots(args.nreal, distribution, configuration, index).as_posix())

    # fine scale part of neumann snapshot
    u_fine = fem.Function(transfer_problem.subproblem.V)
    u_fine.x.array[:] = U_orth.to_numpy().flatten()
    fine_scale_part(u_fine, transfer_problem.subproblem.W, in_place=True)

    # ### Extend basis
    for edge in active_edges:
        dofs = transfer_problem.subproblem.V_to_L[edge]
        u_values = u_fine.x.array[np.newaxis, dofs]
        Lf = transfer_problem.subproblem.edge_spaces["fine"][edge]
        rs = FenicsxVectorSpace(Lf)
        neumann_mode = rs.from_numpy(u_values)

        extend_basis(
            neumann_mode,
            edge_bases[edge],
            product=range_products[edge],
            method="gram_schmidt",
        )

    table_basis_length = []
    table_basis_length.append(["Edge", "Basis length"])
    for edge, rb in edge_bases.items():
        table_basis_length.append([edge, len(rb)])
    table_title = f"\nNumber of basis functions after adding Neumann mode ({pid=})."
    logger.info(format_table(table_basis_length, title=table_title))

    # ### Conversion to picklable objects
    # passing data along processes requires picklable data
    # fenics stuff is not picklable ...
    bases = {}
    products = {}
    for edge, basis in edge_bases.items():
        cop = NumpyConversionOperator(basis.space, direction="to_numpy")
        bases[edge] = cop.apply(basis)

        product_matrix = csr_array(range_products[edge].matrix.getValuesCSR()[::-1])
        product = NumpyMatrixOperator(
            product_matrix,
            source_id=bases[edge].space.id,
            range_id=bases[edge].space.id,
        )
        products[edge] = product

    return bases, products


def main(args):
    from .tasks import example

    method = Path(__file__).stem  # hapod
    logfilename = example.log_edge_basis(
        args.nreal, method, args.distribution, args.configuration
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level="INFO")

    # The training set was already defined in order to generate physical meshes.
    # Here, the actual parameter values are not needed, but for each mesh I have
    # to run the range finder.

    ntrain = example.ntrain(args.configuration)
    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {ntrain}."
    )

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

    rrf_bases_length = defaultdict(list)
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # first sample
    index = 0
    ss = seed_seqs_rrf[index]
    with new_rng(ss):
        bases, range_products = approximate_range(
            args, example, index, logfilename
        )

    snapshots = {}
    for edge, basis in bases.items():
        snapshots[edge] = basis.space.empty()
        snapshots[edge].append(basis)
        rrf_bases_length[edge].append(len(basis))

    for index in range(1, ntrain):
        ss = seed_seqs_rrf[index]
        with new_rng(ss):
            bases, _ = approximate_range(
                args, example, index, logfilename
            )
        for edge, rb in bases.items():
            snapshots[edge].append(rb)
            rrf_bases_length[edge].append(len(rb))

    pod_table = []
    pod_table.append(
        ["Edge", "Number of Snapshots", "Number of POD modes", "Rel. Tolerance"]
    )
    pod_modes = {}
    pod_svals = {}
    pod_data = {}

    for edge, snapshot_data in snapshots.items():
        # ---- ensemble mean subtraction ----
        num_snapshots = len(snapshot_data)
        snaps = snapshot_data.copy()
        snaps.scal(1. / num_snapshots)
        ensemble_mean = sum(snaps)
        snapshot_data.axpy(-1., ensemble_mean)
        # ---- ensemble mean subtraction ----
        modes, svals = pod(
            snapshot_data,
            product=range_products[edge],
            modes=None,
            rtol=example.pod_rtol,
        )
        pod_modes[edge] = modes.to_numpy()
        pod_svals[edge] = svals
        pod_table.append([edge, len(snapshot_data), len(modes), example.pod_rtol])
        pod_data[edge] = (len(modes), len(snapshot_data))

    logger.info(format_table(pod_table, title="\nPOD of edge basis functions"))

    # write output: fine scale edge modes, singular values, rrf bases length
    np.savez(
        example.fine_scale_edge_modes_npz(
            args.nreal, method, args.distribution, args.configuration
        ),
        **pod_modes,
    )
    np.savez(
        example.hapod_singular_values_npz(
            args.nreal, args.distribution, args.configuration
        ),
        **pod_svals,
    )
    np.savez(
        example.rrf_bases_length(
            args.nreal, method, args.distribution, args.configuration
        ),
        **rrf_bases_length,
    )
    with example.hapod_pod_data(args.nreal, args.distribution, args.configuration).open(
        "w"
    ) as fh:
        fh.write(json.dumps(pod_data))


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
