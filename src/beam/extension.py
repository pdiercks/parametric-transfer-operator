from pathlib import Path
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from basix.ufl import element
import numpy as np

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from pymor.tools.table import format_table

from multi.bcs import BoundaryDataFactory
from multi.basis_construction import compute_phi
from multi.domain import RectangularSubdomain
from multi.dofmap import QuadrilateralDofLayout
from multi.extension import extend
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.interpolation import make_mapping


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem

    # ### logger
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_extension(
                args.distribution, args.name, args.cell
            )
        }
    )
    logger = getLogger(Path(__file__).stem, level="INFO")

    # problem definition
    beamproblem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    coarsegrid = beamproblem.coarse_grid

    # ### Subdomain problem
    # TODO translate the subdomain to correct global coordinates
    # this will make postprocessing easier?
    gdim = beam.gdim
    domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()
    degree = beam.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)
    problem = LinElaSubProblem(omega, V, phases=(mat,))
    problem.setup_coarse_space()
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    boundary_entities = np.array([], dtype=np.intc)
    edges = set(["bottom", "left", "right", "top"])
    for edge in edges:
        edge_entities = mesh.locate_entities_boundary(
            problem.domain.grid,
            problem.domain.tdim - 1,
            problem.domain.str_to_marker(edge),
        )
        boundary_entities = np.append(boundary_entities, edge_entities)

    bc_factory = BoundaryDataFactory(problem.domain.grid, boundary_entities, problem.V)
    zero_function = fem.Function(problem.V)
    zero_function.x.array[:] = 0.0
    boundary_data = list()

    mask = {}
    start = 0
    end = 0

    dof_layout = QuadrilateralDofLayout()
    cell_edges = coarsegrid.get_entities(1, args.cell)

    table = []
    table.append(["Global edge index", "Owning cell", "Local edge"])

    for local_ent, edge in enumerate(cell_edges):
        (ci, loc_edge) = beamproblem.edge_to_cell(edge)
        table.append([edge, ci, loc_edge])

        # ### Fine scale edge modes
        configuration = beam.cell_to_config(ci)
        infile = beam.fine_scale_edge_modes_npz(args.distribution, configuration, args.name)
        logger.debug(
            f"Reading fine scale modes for cell {args.cell} from file: {infile}"
        )
        fine_scale_edge_modes = np.load(infile)

        modes = fine_scale_edge_modes[loc_edge]
        logger.debug(
            f"Number of fine scale modes for local edge {loc_edge}: {len(modes)}"
        )
        end += modes.shape[0]

        if args.cell == ci:
            # args.cell owns loc_edge
            boundary = loc_edge
        else:
            # args.cell does not own loc_edge
            # in this case modes from neighbouring configuration are extended
            # the mapping of DOFs between different edge spaces has to be considered
            boundary = dof_layout.local_edge_index_map[local_ent]
            map = make_mapping(
                problem.edge_spaces["fine"][boundary],
                problem.edge_spaces["fine"][loc_edge],
            )
            modes = modes[:, map]

        mask[boundary] = np.s_[start:end]
        start += modes.shape[0]

        dofs = problem.V_to_L[boundary]
        zero_boundaries = list(edges.difference(set([boundary])))

        for mode in modes:
            bc = []
            g = bc_factory.create_function_values(mode, dofs)
            bc.append(
                {
                    "value": g,
                    "boundary": problem.domain.str_to_marker(boundary),
                    "method": "geometrical",
                }
            )
            for gamma_0 in zero_boundaries:
                bc.append(
                    {
                        "value": zero_function,
                        "boundary": problem.domain.str_to_marker(gamma_0),
                        "method": "geometrical",
                    }
                )
            assert len(bc) == 4
            boundary_data.append(bc)
    logger.debug(f"Total number of extensions: {len(boundary_data)}")

    table_title = f"Cell index: {args.cell}."
    logger.info(format_table(table, title=table_title))

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    extensions = extend(
        problem,
        boundary_entities,
        boundary_data=boundary_data,
        petsc_options=petsc_options,
    )
    source = FenicsxVectorSpace(V)
    U = source.make_array(extensions)
    chi = {}
    for edge, view in mask.items():
        chi[edge] = U[view].to_numpy()
        logger.info(f"Number of fine scale modes for {edge=}: {len(chi[edge])}.")

    # ### Read coarse scale basis
    vertices = omega.coarse_grid.topology.connectivity(2, 0).links(0)
    x_vertices = mesh.compute_midpoints(omega.coarse_grid, 0, vertices)
    phi_vectors = compute_phi(problem, x_vertices)
    phi = source.make_array(phi_vectors)

    # ### Write full basis
    full_basis = source.empty()
    full_basis.append(phi)
    full_basis.append(U)
    viz = FenicsxVisualizer(source)
    viz.visualize(
        full_basis,
        filename=beam.fine_scale_modes_bp(args.distribution, args.name, args.cell),
    )

    # ### Write complete basis to single file
    np.savez(
        beam.local_basis_npz(args.distribution, args.name, args.cell),
        phi=phi.to_numpy(),
        **chi,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Extend fine scale edge modes and write local basis for given coarse grid cell (subdomain)"
    )
    parser.add_argument(
        "distribution",
        type=str,
        help="The distribution used in the range approximation.",
        choices=("normal", "multivariate_normal"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument("cell", type=int, help="The coarse grid cell index.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
