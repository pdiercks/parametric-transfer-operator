import pathlib
import numpy as np

import dolfinx as df
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from multi.problems import LinElaSubProblem
from multi.domain import RectangularSubdomain


def unit_cell_boundary(domain):
    """Returns all entities of the unit cell boundary"""
    boundary_entities = np.array([], dtype=np.intc)
    edges = set(["bottom", "left", "right", "top"])
    for edge in edges:
        edge_entities = df.mesh.locate_entities_boundary(
            domain.grid,
            domain.tdim - 1,
            domain.str_to_marker(edge),
        )
        boundary_entities = np.append(boundary_entities, edge_entities)
    return boundary_entities


def discretize_subdomain_operator(args, example):
    from .auxiliary_problem import discretize_auxiliary_problem
    from .matrix_based_operator import FenicsxMatrixBasedOperator
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
    mesh = aux.problem.domain.grid
    facettags = aux.problem.domain.facet_tags
    celltags = aux.problem.domain.cell_tags
    omega = RectangularSubdomain(args.cell, mesh, cell_tags=celltags, facet_tags=facettags)
    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d) # type: ignore

    # ### wrap stiffness matrix as pymor operator
    def param_setter(mu):
        d.x.array[:] = 0.0 # type: ignore
        aux.solve(d, mu) # type: ignore
        d.x.scatter_forward() # type: ignore

    # TODO
    # Dirichlet bcs for extension?

    # TODO
    # for each parameter mu
    # need to factorize the matrix
    # need to use the compiled form to define rhs (Dirichlet lift)
    # (this is easier if it is a problem that knows about both a(u, v) and L(v))

    params = example.parameters["subdomain"]
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs, params, param_setter=param_setter, name="ParaGeom"
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
    logger = getLogger(pathlib.Path(__file__).stem, level="INFO")

    # problem definition
    beamproblem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    coarsegrid = beamproblem.coarse_grid

    # ### Subdomain Discretization
    operator, parageom = discretize_subdomain_operator(args, example)
    subdomain = parageom.domain
    V = parageom.V
    material = parageom.mat

    subdomain.create_coarse_grid(1)
    subdomain.create_boundary_grids()

    problem = LinElaSubProblem(subdomain, V, phases=material)
    problem.setup_coarse_space()
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()
    problem.create_edge_space_maps()

    boundary = unit_cell_boundary(subdomain)
    breakpoint()

    # define training set
    # for each mu
    # discretize operator (lhs)
    # gather data for extension (separate for each edge and coarse scale mode)
    # extend ; store extended modes as snapshots (e.g. all bottom modes extended for all training parameter values)
    # num_snapshots = num_{edge}_modes * num_train

    # do POD over each snapshot set


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
    args = parser.parse_args(sys.argv[1:])
    main(args)
