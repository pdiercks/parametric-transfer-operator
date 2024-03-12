"""approximate the range of transfer operators for fixed parameter values"""

import os
from typing import Optional
# from itertools import repeat
from pathlib import Path
# import concurrent.futures
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv
from time import perf_counter

from dolfinx import mesh, fem, default_scalar_type

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import NumpyConversionOperator
from pymor.parameters.base import Parameters, ParameterSpace
from pymor.tools.random import new_rng
from pymor.reductors.basic import extend_basis
from pymor.tools.table import format_table

from multi.misc import x_dofs_vectorspace
from multi.problems import TransferProblem
from multi.sampling import correlation_matrix, create_random_values
from multi.projection import orthogonal_part, fine_scale_part
from multi.dofmap import QuadrilateralDofLayout
from multi.product import InnerProduct
from multi.shapes import NumpyLine


def adaptive_edge_rrf_normal(
    transfer_problem: TransferProblem,
    active_edges,
    source_product: Optional[Operator] = None,
    range_product: Optional[str] = None,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min = None,
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

    logger = getLogger("adaptive_edge_rrf", level="DEBUG")
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
            (source_product.source.dim, source_product.range.dim), # type: ignore
            matvec=mv,  # type: ignore
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim), # type: ignore
            matvec=mvinv,  # type: ignore
        )
        lambda_min = eigsh(
            L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
        )[0]

    # ### test set
    R = tp.generate_random_boundary_data(
        count=num_testvecs, distribution=distribution
    )
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

    num_solves = 0
    while np.any(maxnorm > testlimit):
        v = tp.generate_random_boundary_data(
            1, distribution, **sampling_options
        )

        U = tp.solve(v)
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

        logger.info(f"{maxnorm=}")

    return pod_bases, range_products, num_solves


def build_mvn_training_set(transfer_problem):
    xmin = transfer_problem.problem.domain.xmin
    xmax = transfer_problem.problem.domain.xmax
    L_corr = np.linalg.norm(xmax - xmin).item() * 3

    x_dofs = x_dofs_vectorspace(transfer_problem.problem.V)
    points = x_dofs[transfer_problem.bc_dofs_gamma_out]

    num_samples = 0
    training_set = []
    while True:
        sigma = correlation_matrix(points, L_corr)
        print(f"Build Sigma of shape {sigma.shape} for {L_corr=}.")
        λ_max = eigsh(sigma, k=1, which="LM", return_eigenvectors=False)
        rtol = 5e-3
        eigvals = eigh(
            sigma,
            eigvals_only=True,
            driver="evx",
            subset_by_value=[λ_max.item() * rtol, np.inf],
        )
        num_eigvals = eigvals.size
        print(f"Found {num_eigvals=}.")

        inc = num_eigvals - num_samples
        if inc > 0:
            mean = np.zeros(sigma.shape[0])
            u = create_random_values(
                (inc, sigma.shape[0]),
                distribution="multivariate_normal",
                mean=mean,
                cov=sigma,
                method="eigh",
            )
            training_set.append(u)

            num_samples += inc
            print(f"Added {inc=} samples")
            print("Decreasing correlation length ...")
            L_corr /= 2
        # elif inc == 0:
        #     # this means the correlation length was not decreased enough
        #     # but might happen for large L_corr in the beginning
        #     # How to differentiate between the above case and case when L_corr is already sufficiently small?
        #     L_corr /= 2
        else:
            break

    mvn_train_set = np.vstack(training_set)
    print(f"Build mvn training set of size {len(mvn_train_set)}")
    return mvn_train_set


def approximate_range(beam, mu, configuration, distribution="normal"):
    from .definitions import BeamProblem
    from .locmor import discretize_oversampling_problem

    logger = getLogger(
        "range_approximation",
        level="INFO",
        filename=beam.log_edge_range_approximation(distribution, configuration).as_posix(),
    )
    pid = os.getpid()
    logger.info(f"{pid=},\tApproximating range of T for {mu=} using {distribution=}.\n")

    tic = perf_counter()
    transfer_problem = discretize_oversampling_problem(beam, mu, configuration)
    logger.info(f"{pid=},\tDiscretized transfer problem in {perf_counter()-tic}.")

    if distribution == "normal":
        sampling_options = {}
    elif distribution == "multivariate_normal":
        mvn_train_set = build_mvn_training_set(transfer_problem)
        sampling_options = {"training_set": mvn_train_set}
    else:
        raise NotImplementedError

    # Multiscale problem definition
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    cell_index = beam_problem.config_to_cell(configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    dirichlet = beam_problem.get_dirichlet(cell_index)
    # determine basis for each edge
    # in contrast to beam_problem.active_edges(cell_index)
    active_edges = set(["bottom", "left", "right", "top"])

    ttol = beam.rrf_ttol
    ftol = beam.rrf_ftol
    num_testvecs = beam.rrf_num_testvecs
    source_product = transfer_problem.source_product
    range_product = beam.range_product

    # TODO approximate range in context of new_rng (if number of realizations > 1)
    edge_bases, range_products, num_solves = adaptive_edge_rrf_normal(
        transfer_problem,
        active_edges,
        source_product=source_product,
        range_product=range_product,
        error_tol=ttol,
        failure_tolerance=ftol,
        num_testvecs=num_testvecs,
    )
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
    loading = fem.Constant(
        omega.grid, (default_scalar_type(0.0), default_scalar_type(-10.0))
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
            u_in.function_space.mesh._cpp_object,
            u_in.function_space.element,
            u_neumann.function_space.mesh._cpp_object,
        ),
    )

    U_in = transfer_problem.range.make_array([u_in.vector])
    if transfer_problem.kernel is None:
        U_orth = U_in
    else:
        U_orth = orthogonal_part(
            U_in, transfer_problem.kernel, product=transfer_problem.range_product, orthonormal=True
        )

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
            product_matrix, source_id=bases[edge].space.id, range_id=bases[edge].space.id
        )
        products[edge] = product

    return bases, products


def main(args):
    from .tasks import beam
    from .lhs import sample_lhs

    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_edge_range_approximation(
                args.distribution, args.configuration, "hapod"
            ).as_posix(),
        }
    )
    logger = getLogger(Path(__file__).stem, level="INFO")

    sampling_options = beam.lhs_options[args.configuration]
    logger.debug(f"{sampling_options=}")

    # TODO simply do serial version to just test if correlated random samples
    # are actually the reason for the quality of the basis
    realizations = np.load(beam.realizations)
    lhs_seed = np.random.SeedSequence(realizations[0]).generate_state(1)
    # TODO add real as command line argument

    mu_name = sampling_options.get("name")
    ndim = sampling_options.get("ndim")
    samples = sampling_options.get("samples")
    criterion = sampling_options.get("criterion")
    # seed sequences for the randomized range finder for each transfer operator
    seed_seqs_rrf = np.random.SeedSequence(realizations[0]).generate_state(samples+1)[1:]

    param = Parameters({mu_name: ndim})
    parameter_space = ParameterSpace(param, beam.mu_range)
    training_set = sample_lhs(
        parameter_space,
        mu_name,
        samples=samples,
        criterion=criterion,
        random_state=lhs_seed,
    )

    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
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



    # first sample
    mu = training_set[0]
    ss = seed_seqs_rrf[0]
    with new_rng(ss):
        bases, range_products = approximate_range(
                beam, mu, args.configuration, args.distribution
                )

    snapshots = {}
    for edge, basis in bases.items():
        snapshots[edge] = basis.space.empty()
        snapshots[edge].append(basis)

    for mu, ss in zip(training_set[1:], seed_seqs_rrf[1:]):
        with new_rng(ss):
            bases, _ = approximate_range(
                beam, mu, args.configuration, args.distribution
            )
        for edge, rb in bases.items():
            snapshots[edge].append(rb)

    # gather edge bases (for each edge)
    # do POD for each of the edges
    # save final pod modes and singular values

    # no need to reinstantiate fenics objects
    # direclty save edge basis functions for each edge
    # as input to extension task

    # snapshots = range_product.range.empty()
    # for b in snaps:
    #     snapshots.append(b)

    # basis, range_product = next(results)
    # snapshots = range_product.range.empty()
    # snapshots.append(basis)
    # for rb, _ in results:
    #     snapshots.append(rb)

    # num_snapshots = [len(basis) for basis in snaps]
    # num_modes = int(np.mean(num_snapshots) * 2)
    pod_table = []
    pod_table.append(["Edge", "Number of Snapshots", "Number of POD modes", "Rel. Tolerance"])
    pod_modes = {}
    pod_svals = {}
    for edge, snapshot_data in snapshots.items():
        modes, svals = pod(snapshot_data, product=range_products[edge], modes=None, rtol=beam.pod_rtol)
        pod_modes[edge] = modes.to_numpy()
        pod_svals[edge] = svals
        pod_table.append([edge, len(snapshot_data), len(modes), beam.pod_rtol])
    logger.info(format_table(pod_table, title="\nPOD of edge basis functions"))

    # write pod modes and singular values to disk
    np.savez(
        beam.fine_scale_edge_modes_npz(args.distribution, args.configuration, "hapod"), **pod_modes
    )
    np.savez(beam.loc_singular_values_npz(args.distribution, args.configuration), **pod_svals)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument(
        "--max_workers", type=int, default=4, help="The max number of workers."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
