import pathlib
import numpy as np

import dolfinx as df
from petsc4py import PETSc
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.algorithms.pod import pod
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVisualizer
from multi.problems import LinElaSubProblem
from multi.domain import RectangularSubdomain
from multi.bcs import BoundaryDataFactory
from multi.dofmap import QuadrilateralDofLayout
from multi.product import InnerProduct
from multi.shapes import NumpyQuad


def unit_cell_boundary(domain):
    """Returns all entities of the unit cell boundary"""
    boundary_entities = np.array([], dtype=np.intc)
    entity_dim = domain.tdim - 1
    edges = set(["bottom", "left", "right", "top"])
    for edge in edges:
        edge_entities = df.mesh.locate_entities_boundary(
            domain.grid,
            entity_dim,
            domain.str_to_marker(edge),
        )
        boundary_entities = np.append(boundary_entities, edge_entities)
    return boundary_entities, entity_dim


def discretize_subdomain_operator(args, example):
    from .auxiliary_problem import discretize_auxiliary_problem
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCTopo
    from .fom import ParaGeomLinEla

    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, ftags, example.parameters["subdomain"]
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    EMOD = example.youngs_modulus
    POISSON = example.poisson_ratio

    # re-instantiate domain as RectangularSubdomain
    mesh = aux.problem.domain.grid
    facettags = aux.problem.domain.facet_tags
    celltags = aux.problem.domain.cell_tags
    omega = RectangularSubdomain(
        args.cell, mesh, cell_tags=celltags, facet_tags=facettags
    )

    boundary_entities, entity_dim = unit_cell_boundary(omega)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()

    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d)  # type: ignore
    problem.setup_solver() # create matrix and vector objects

    # ### wrap stiffness matrix as pymor operator
    def param_setter(mu):
        d.x.array[:] = 0.0  # type: ignore
        aux.solve(d, mu)  # type: ignore
        d.x.scatter_forward()  # type: ignore

    # Dirichlet bc to prepare operator for mode extension
    null = df.default_scalar_type(0.0)
    u_zero = df.fem.Constant(omega.grid, (null,) * omega.gdim)
    bc_zero = BCTopo(u_zero, boundary_entities, entity_dim, aux.problem.V)

    # TODO
    # for each parameter mu
    # need to factorize the matrix
    # need to use the compiled form to define rhs (Dirichlet lift)

    params = example.parameters["subdomain"]
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs,
        params,
        param_setter=param_setter,
        bcs=(bc_zero,),
        name="ParaGeom",
    )
    # operator.compiled_form can be used for Dirichlet lift

    return operator, problem


def main(args):
    from .tasks import example
    from .definitions import BeamProblem

    # ### logger
    logfilename = example.log_extension(
        args.nreal, args.method, args.distribution, args.cell
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(pathlib.Path(__file__).stem, level="DEBUG")

    # problem definition
    beamproblem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    coarse_grid = beamproblem.coarse_grid

    # ### Subdomain Discretization
    logger.info("Discretizing subdomain operator ...")
    operator, parageom = discretize_subdomain_operator(args, example)
    V = parageom.V
    material = parageom.mat

    problem = LinElaSubProblem(parageom.domain, V, phases=material)  # type: ignore
    problem.setup_coarse_space()
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()
    problem.create_edge_space_maps()

    bc_topo = operator._bcs[0]
    boundary_entities = bc_topo.entities  # type: ignore
    bc_factory = BoundaryDataFactory(problem.domain.grid, boundary_entities, problem.V)
    zero_function = df.fem.Function(problem.V)
    zero_function.x.array[:] = 0.0  # type: ignore

    dof_layout = QuadrilateralDofLayout()
    cell_edges = coarse_grid.get_entities(1, args.cell)
    edges = set(["bottom", "left", "right", "top"])

    # do POD over each snapshot set
    parameter_space = operator.parameters.space(example.mu_range)
    training_set = parameter_space.sample_uniformly(args.ntrain)

    def apply_dirichlet_lift(b, L, a, bcs: list):
        with b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(b, L)

        # Apply boundary conditions to the rhs
        apply_lifting(b, [a], bcs=[bcs])  # type: ignore
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        set_bc(b, bcs)

    assert not parageom._bc_handler.has_neumann
    rhs = parageom.b
    L = parageom.L
    a = operator.compiled_form

    snapshots = {
            "bottom": operator.source.empty(),
            "left": operator.source.empty(),
            "right": operator.source.empty(),
            "top": operator.source.empty()
            }
    num_snapshots = {}

    for mu in training_set:
        logger.info(f"Assembly of matrix operator for {mu=}.")
        matrix_op = operator.assemble(mu)  # updates transformation displacement

        for local_ent, edge_tag in enumerate(cell_edges):
            edge = dof_layout.local_edge_index_map[local_ent]
            logger.debug(f"{edge=}")
            logger.debug(f"{local_ent=}")
            logger.debug(f"{edge_tag=}")
            (ci, loc_edge) = beamproblem.edge_to_cell(edge_tag)
            logger.debug(f"{loc_edge=}")
            configuration = example.cell_to_config(ci)
            instream = example.fine_scale_edge_modes_npz(
                args.nreal, args.method, args.distribution, configuration
            )
            logger.debug(
                f"Reading fine scale modes for cell {args.cell} from file: {instream}"
            )
            fine_scale_edge_modes = np.load(instream)
            modes = fine_scale_edge_modes[loc_edge]  # all modes for some edge
            logger.debug(
                f"Number of fine scale modes for local edge {loc_edge}: {len(modes)}"
            )
            num_snapshots[edge] = len(modes) * args.ntrain
            logger.debug(f"{num_snapshots[edge]=} for {edge=}")

            if args.cell == ci:
                # args.cell owns loc_edge
                boundary = loc_edge
            else:
                # args.cell does not own loc_edge
                # in this case modes from neighbouring configuration are extended
                # the mapping of DOFs between different edge spaces has to be considered
                # map from `loc_edge` to `boundary`
                boundary = dof_layout.local_edge_index_map[local_ent]
                if loc_edge == "left":
                    map = problem.edge_space_maps["left_to_right"]
                    assert boundary == "right"
                elif loc_edge == "right":
                    map = problem.edge_space_maps["right_to_left"]
                    assert boundary == "left"
                elif loc_edge == "top":
                    map = problem.edge_space_maps["top_to_bottom"]
                    assert boundary == "bottom"
                elif loc_edge == "bottom":
                    map = problem.edge_space_maps["bottom_to_top"]
                    assert boundary == "top"
                else:
                    raise NotImplementedError
                modes = modes[:, map]

            dofs = problem.V_to_L[boundary]
            zero_boundaries = list(edges.difference(set([boundary])))

            # ### Assemble rhs for each mode
            vectors = []

            for mode in modes:
                problem.clear_bcs()
                rhs.zeroEntries()
                g = bc_factory.create_function_values(mode, dofs)
                problem.add_dirichlet_bc(
                    g,
                    boundary=problem.domain.str_to_marker(boundary), # type: ignore
                    method="geometrical",
                )
                for gamma_0 in zero_boundaries:
                    problem.add_dirichlet_bc(
                        zero_function,
                        boundary=problem.domain.str_to_marker(gamma_0), # type: ignore
                        method="geometrical",
                    )

                bcs = problem.get_dirichlet_bcs()
                apply_dirichlet_lift(rhs, L, a, bcs)
                vectors.append(rhs.copy())

            # ### Extension
            lifting = matrix_op.range.make_array(vectors)  # type: ignore
            logger.info(
                f"Computing extension of {len(lifting)} modes for {mu=} and edge = {loc_edge}"
            )
            extensions = matrix_op.apply_inverse(lifting)
            snapshots[edge].append(extensions)

    inner_product = InnerProduct(problem.V, "h1")
    product_mat = inner_product.assemble_matrix()
    h1_product = FenicsxMatrixOperator(product_mat, problem.V, problem.V)
    viz = FenicsxVisualizer(operator.source)

    basis = {}
    svals = {}

    for edge in edges:
        logger.info(f"Computing POD of extensions for {edge=}.")
        pod_modes, pod_svals = pod(snapshots[edge], product=h1_product) # type: ignore
        basis[edge] = pod_modes
        svals[edge] = pod_svals

    # TODO: make this a function? or at least move up, such that
    # matrix_op = operator.assemble(mu) is only done once per mu

    cell_vertices = parageom.domain.coarse_grid.topology.connectivity(2, 0).links(0)
    nodes = df.mesh.compute_midpoints(parageom.domain.coarse_grid, 0, cell_vertices)
    quadrilateral = NumpyQuad(nodes)
    shape_functions = quadrilateral.interpolate(operator.source.V)
    phi = df.fem.Function(operator.source.V)

    phi_snapshots = {}

    for k, shape in enumerate(shape_functions):
        phi_snapshots[k] = operator.source.empty(reserve=args.ntrain)
        problem.clear_bcs()
        rhs.zeroEntries()

        phi.x.array[:] = shape
        bcs = [bc_factory.create_bc(phi.copy())]
        apply_dirichlet_lift(rhs, L, a, bcs)
        lifting = operator.range.make_array([rhs.copy()])

        for mu in training_set:
            matrix_op = operator.assemble(mu)
            ext = matrix_op.apply_inverse(lifting)
            phi_snapshots[k].append(ext)

    # FIXME
    # for the POD over phi
    # need to make sure that functions are still bilinear on the boundary ...
    inner_product = InnerProduct(problem.V, "h1", operator.bcs)
    mat = inner_product.assemble_matrix()
    h1_0_product = FenicsxMatrixOperator(mat, problem.V, problem.V)
    phi_0, svals_0 = pod(phi_snapshots[0], product=h1_0_product)
    viz.visualize(phi_0, filename="phi_0.xdmf")
    breakpoint()



if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nreal", type=int, help="The `nreal`-th realization of the problem."
    )
    parser.add_argument(
        "method",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod",),
    )
    parser.add_argument(
        "distribution",
        type=str,
        help="The distribution used in the range approximation.",
        choices=("normal",),
    )
    parser.add_argument("cell", type=int, help="The coarse grid cell index.")
    parser.add_argument("ntrain", type=int, help="Size of the training set.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
