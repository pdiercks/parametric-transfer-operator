import numpy as np
from pathlib import Path

from dolfinx import mesh, fem, default_scalar_type

from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.tools.random import new_rng
from pymor.parameters.base import Parameters, ParameterSpace
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.reductors.basic import extend_basis

from multi.dofmap import QuadrilateralDofLayout
from multi.product import InnerProduct
from multi.shapes import NumpyLine
from multi.projection import orthogonal_part, fine_scale_part


def lhs_design(beam, sampling_options):
    from .lhs import sample_lhs

    mu_name = sampling_options.get("name")
    ndim = sampling_options.get("ndim")
    samples = sampling_options.get("samples")
    criterion = sampling_options.get("criterion")
    random_state = sampling_options.get("random_state")

    param = Parameters({mu_name: ndim})
    parameter_space = ParameterSpace(param, beam.mu_range)
    training_set = sample_lhs(
        parameter_space,
        mu_name,
        samples=samples,
        criterion=criterion,
        random_state=random_state,
    )
    return training_set


def solve_neumann_problem(beam, beam_problem, transfer_problem, mu, args, neumann_snapshots):
    neumann_problem = transfer_problem.problem
    neumann_problem.clear_bcs()
    omega = neumann_problem.domain

    emod_base = beam.youngs_modulus
    num_phases = len(transfer_problem.problem.phases)
    material = tuple([dict(E=1.0, NU=0.3) for _ in range(num_phases)])
    mu_values = mu.to_numpy()
    for mu_i, phase in zip(mu_values, material):
        phase.update({"E": emod_base * mu_i, "NU": 0.3})

    transfer_problem.update_material(material)

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
    configuration = args.configuration
    cell_index = beam_problem.config_to_cell(configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    neumann_problem.add_dirichlet_bc(zero, gamma_out, method="geometrical")

    # Add homogeneous Dirichlet BCs if present
    dirichlet = beam_problem.get_dirichlet(cell_index)
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
    active_edges = set(["bottom", "left", "right", "top"])
    for edge in active_edges:
        dofs = transfer_problem.subproblem.V_to_L[edge]
        u_values = u_fine.x.array[np.newaxis, dofs]
        Lf = transfer_problem.subproblem.edge_spaces["fine"][edge]
        rs = FenicsxVectorSpace(Lf)
        neumann_mode = rs.from_numpy(u_values)
        neumann_snapshots[edge].append(neumann_mode)

        # extend_basis(
        #         neumann_mode,
        #         edge_bases[edge],
        #         product=range_products[edge],
        #         method="gram_schmidt",
        #         )


def approximate_range(
    logger,
    beam,
    distribution,
    configuration,
    training_set,
    testing_set,
    lambda_min=None,
):
    """heuristic version of the range finder"""
    from .definitions import BeamProblem
    from .locmor import discretize_oversampling_problem

    assert distribution == "normal"

    # pid = os.getpid()
    # logger.info(f"{pid=},\tApproximating range of T for {mu=} using {distribution=}.\n")

    active_edges = set(["bottom", "left", "right", "top"])

    num_testvecs = beam.rrf_num_testvecs

    # ### Initialize test sets
    test_sets = []
    mu_0 = testing_set[0]
    transfer_problem = discretize_oversampling_problem(beam, mu_0, configuration)
    num_phases = len(transfer_problem.problem.phases)
    R = transfer_problem.generate_random_boundary_data(num_testvecs, distribution)
    M = transfer_problem.solve(R)
    test_sets.append(M)
    # initialize source and range product
    source_product = transfer_problem.source_product

    # ### Caculate min eigenvalue of source product matrix
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

    material = tuple([dict(E=1.0, NU=0.3) for _ in range(num_phases)])
    emod_base = beam.youngs_modulus
    for mu in testing_set[1:]:
        mu_values = mu.to_numpy()
        for mu_i, phase in zip(mu_values, material):
            phase.update({"E": emod_base * mu_i, "NU": 0.3})

        logger.debug(f"current parameter value: {mu}")
        logger.debug(f"current material: {material}")

        transfer_problem.update_material(material)
        R = transfer_problem.generate_random_boundary_data(num_testvecs, distribution)
        M = transfer_problem.solve(R)
        test_sets.append(M)

    # subtract coarse scale part for each test set ...
    dof_layout = QuadrilateralDofLayout()

    # ### initialize data structures for edge range finder
    # for each test set M, we have 4 test sets for the fine scale part of each edge
    # the test vectors of each set M should be separated wrt edges, but
    # regarding mu we do not need to keep them separate, because
    # (a) the source product and hence c_est do not depend on mu and
    # (b) we are only interested in the max norm over all testvectors in all sets
    edge_test_sets = {}  # accumulate all test vectors in this dict
    range_spaces = {}
    range_products = {}
    fine_scale_edge_bases = {}
    neumann_snaps = {}

    maxnorm = np.array(
        [], dtype=float
    )  # maxnorm of testvectors (over all test sets) for each edge
    edges = np.array([], dtype=str)
    coarse_basis = {}
    edge_boundary_dofs = {}  # the dofs for vertices on the boundary of the edge

    for edge_index in range(dof_layout.num_entities[1]):
        edge = dof_layout.local_edge_index_map[edge_index]
        edges = np.append(edges, edge)

        edge_mesh = transfer_problem.subproblem.domain.fine_edge_grid[edge]
        edge_space = transfer_problem.subproblem.edge_spaces["fine"][edge]
        range_spaces[edge] = FenicsxVectorSpace(edge_space)

        # ### create dirichletbc for range product
        facet_dim = edge_mesh.topology.dim - 1
        vertices = mesh.locate_entities_boundary(
            edge_mesh, facet_dim, lambda x: np.full(x[0].shape, True, dtype=bool)
        )
        _dofs_ = fem.locate_dofs_topological(edge_space, facet_dim, vertices)
        gdim = transfer_problem.subproblem.domain.grid.geometry.dim
        range_bc = fem.dirichletbc(
            np.array((0,) * gdim, dtype=default_scalar_type), _dofs_, edge_space
        )
        edge_boundary_dofs[edge] = range_bc._cpp_object.dof_indices()[0]

        # ### range product in edge space
        inner_product = InnerProduct(edge_space, beam.range_product, bcs=(range_bc,))
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
        coarse_basis[edge] = range_spaces[edge].from_numpy(shape_funcs)

        # ### edge test sets
        dofs = transfer_problem.subproblem.V_to_L[edge]
        edge_test_sets[edge] = range_spaces[edge].empty()
        for testset in test_sets:
            edge_test_sets[edge].append(
                range_spaces[edge].from_numpy(testset.dofs(dofs))
            )
        # subtract coarse scale part
        coeff = edge_test_sets[edge].dofs(edge_boundary_dofs[edge])
        edge_test_sets[edge] -= coarse_basis[edge].lincomb(coeff)
        assert np.isclose(
            np.sum(edge_test_sets[edge].dofs(edge_boundary_dofs[edge])), 1e-9
        )

        if edge in active_edges:
            maxnorm = np.append(maxnorm, np.inf)
            fine_scale_edge_bases[edge] = range_spaces[edge].empty()
            neumann_snaps[edge] = range_spaces[edge].empty()
        else:
            maxnorm = np.append(maxnorm, 0.0)

    failure_tolerance = beam.rrf_ftol
    target_tolerance = beam.rrf_ttol
    num_source_dofs = len(transfer_problem._bc_dofs_gamma_out)
    testfail = np.array(
        [
            failure_tolerance / min(num_source_dofs, space.dim)
            for space in range_spaces.values()
        ]
    )
    testlimit = (
        np.sqrt(2.0 * lambda_min)
        * erfinv(testfail ** (1.0 / num_testvecs))
        * target_tolerance
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    num_solves = 0
    while np.any(maxnorm > testlimit):
        # select parameter value from training set
        # simply march through LHS design from start to end for now
        mu = training_set[num_solves]

        # update the system matrix
        mu_values = mu.to_numpy()
        for mu_i, phase in zip(mu_values, material):
            phase.update({"E": emod_base * mu_i, "NU": 0.3})
        logger.debug(f"while: current parameter value: {mu}")
        logger.debug(f"while: current material: {material}")
        transfer_problem.update_material(material)

        # solve oversampling problem
        v = transfer_problem.generate_random_boundary_data(1, distribution)
        U = transfer_problem.solve(v)
        num_solves += 1

        target_edges = edges[maxnorm > testlimit]
        for edge in target_edges:
            B = fine_scale_edge_bases[edge]
            edge_space = range_spaces[edge]

            # restrict the training sample to the edge
            udofs = edge_space.from_numpy(U.dofs(transfer_problem.subproblem.V_to_L[edge]))
            coeff = udofs.dofs(edge_boundary_dofs[edge])
            ufine = udofs - coarse_basis[edge].lincomb(coeff)

            # extend basis
            extend_basis(ufine, B, product=range_products[edge], method="gram_schmidt")

            # orthonormalize all test sets wrt current basis
            M = edge_test_sets[edge]
            M -= B.lincomb(B.inner(M, range_products[edge]).T)

            norm = M.norm(range_products[edge])
            maxnorm[dof_layout.local_edge_index_map[edge]] = np.max(norm)

        logger.info(f"Iteration: {num_solves}:\t{maxnorm=}.")

    with logger.block("Completed range approximation"):
        logger.info(f"Number of iterations: {num_solves}")
        for edge in edges:
            logger.info(f"Number of modes {edge}: {len(fine_scale_edge_bases[edge])}")

    # Multiscale problem definition
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    # ### Extend basis by neumann solutions
    for i in range(num_solves):
        mu = training_set[i]
        # FIXME avoid adding more functions if extension fails for i < num_solves
        solve_neumann_problem(beam, beam_problem, transfer_problem, mu, args, neumann_snaps)

    for edge, snapshots in neumann_snaps.items():
        extend_basis(snapshots, fine_scale_edge_bases[edge], product=range_products[edge],
                     method="pod", pod_modes=len(snapshots))

    with logger.block("Completed Neumann data extension"):
        for edge in edges:
            logger.info(f"Number of modes {edge}: {len(fine_scale_edge_bases[edge])}")
    return fine_scale_edge_bases



def main(args):
    from .tasks import beam

    # TODO: add beam.log_heuristic_range_approx
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_edge_range_approximation(
                args.distribution, args.configuration, "heuristic"
            ).as_posix(),
        }
    )
    logger = getLogger(Path(__file__).stem, level="DEBUG")

    realizations = np.load(beam.realizations)
    # FIXME hardcoded to 1 realization
    test_seed, train_seed, rrf_seed = np.random.SeedSequence(
        realizations[0]
    ).generate_state(3)

    # FIXME might need to add more samples here
    # how to select among these samples during the while loop?
    # ideally these would be selected in a greedy fashion
    train_options = beam.lhs_options[args.configuration]
    train_options["random_state"] = train_seed
    training_set = lhs_design(beam, train_options)

    test_options = beam.lhs_options[args.configuration]
    test_options["random_state"] = test_seed
    testing_set = lhs_design(beam, test_options)

    logger.info(
        "Starting heuristic range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
    )

    with new_rng(rrf_seed):
        bases = approximate_range(
            logger,
            beam,
            args.distribution,
            args.configuration,
            training_set,
            testing_set,
        )

    # write fine scale basis
    np.savez(
        beam.fine_scale_edge_modes_npz(args.distribution, args.configuration, "heuristic"),
        **bases,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Approximate the range of a parametric transfer operator using the heuristic rrf",
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
    args = parser.parse_args(sys.argv[1:])
    main(args)
