from pathlib import Path
from collections import defaultdict

from scipy.sparse import coo_array
import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from pymor.parameters.base import Parameters
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.projection import project
from pymor.models.basic import StationaryModel
from pymor.tools.random import new_rng

from multi.interpolation import make_mapping
from multi.boundary import point_at
from multi.domain import RectangularSubdomain
from multi.dofmap import DofMap
from multi.io import BasesLoader, select_modes
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem


# FIXME: this reconstruction method does not seem to be stable wrt the mesh
# it did work using `create_rectangle` but not for `create_voided_rectangle`
# my best guess is that the submesh creation is not reproducible?
# and therefore you cannot assume the same doflayout for every subspace?
# def reconstruct(
#     U_rb: np.ndarray,
#     dofmap: DofMap,
#     bases: list[np.ndarray],
#     subspaces: list[fem.FunctionSpaceBase],
#     cell_maps: list[np.ndarray],
#     u_global: fem.Function,
# ) -> None:
#     """Reconstructs rom solution on the global domain.
#
#     Args:
#         Urb: ROM solution in the reduced space.
#         dofmap: The dofmap of the reduced space.
#         bases: Local basis for each subdomain.
#         subspaces: The subspace of each subdomain.
#         cell_maps: Mapping from child (submesh cell) to parent cell.
#         subdomains: The cells of each subdomain.
#         u_global: The global solution field to be filled with values.
#
#     """
#     V = u_global.function_space
#     u_view = u_global.x.array  # type: ignore
#
#     for i_subdomain, V_sub in enumerate(subspaces):
#         cmap = cell_maps[i_subdomain]
#         u_local = fem.Function(V_sub)
#         u_local_view = u_local.x.array  # type: ignore
#
#         # fill u_local with rom solution
#         basis = bases[i_subdomain]
#         dofs = dofmap.cell_dofs(i_subdomain)
#         u_local_view[:] = U_rb[0, dofs] @ basis
#
#         # fill global u
#         submesh = V_sub.mesh
#         num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
#         for cell in range(num_sub_cells):
#             child_dofs = V_sub.dofmap.cell_dofs(cell)
#             parent_dofs = V.dofmap.cell_dofs(cmap[cell])
#             for parent, child in zip(parent_dofs, child_dofs):
#                 for b in range(V_sub.dofmap.bs):
#                     u_view[parent * V.dofmap.bs + b] = u_local_view[
#                         child * V_sub.dofmap.bs + b
#                     ]


def reconstruct(
    U_rb: np.ndarray,
    dofmap: DofMap,
    bases: list[np.ndarray],
    u_local: fem.Function,
    u_global: fem.Function,
) -> None:
    """Reconstructs rom solution on the global domain.

    Args:
        Urb: ROM solution in the reduced space.
        dofmap: The dofmap of the reduced space.
        bases: Local basis for each subdomain.
        u_local: The local solution field.
        u_global: The global solution field to be filled with values.

    """
    coarse_grid = dofmap.grid
    V = u_global.function_space
    Vsub = u_local.function_space
    submesh = Vsub.mesh
    x_submesh = submesh.geometry.x
    u_global_view = u_global.x.array
    # u_local_view = u_local.x.array

    for cell in range(dofmap.num_cells):

        # translate subdomain mesh
        vertices = coarse_grid.get_entities(0, cell)
        dx_cell = coarse_grid.get_entity_coordinates(0, vertices)[0]
        x_submesh += dx_cell

        # fill u_local with rom solution
        basis = bases[cell]
        dofs = dofmap.cell_dofs(cell)
        # u_local_view[:] = U_rb[0, dofs] @ basis

        # fill global field via dof mapping
        V_to_Vsub = make_mapping(Vsub, V)
        u_global_view[V_to_Vsub] = U_rb[0, dofs] @ basis

        # move subdomain mesh to origin
        x_submesh -= dx_cell
    u_global.x.scatter_forward()


def assemble_system(
    num_modes: int,
    dofmap: DofMap,
    A: FenicsxMatrixOperator,
    b: VectorOperator,
    bases: list[np.ndarray],
    num_max_modes: np.ndarray,
    parameters: Parameters
):
    """Assembles ``operator`` and ``rhs`` for localized ROM as ``StationaryModel``.

    Args:
        num_modes: Number of fine scale modes per edge to be used.
        dofmap: The dofmap of the global reduced space.
        A: Local high fidelity stiffness matrix.
        b: Local high fidelity external force vector.
        bases: Local reduced basis for each subdomain.
        num_max_modes: Maximum number of fine scale modes for each edge.
        parameters: The |Parameters| the ROM depends on.

    """
    from .locmor import COOMatrixOperator

    dofs_per_vertex = 2
    dofs_per_face = 0

    dofs_per_edge = num_max_modes.copy()
    dofs_per_edge[num_max_modes > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vertex, dofs_per_edge, dofs_per_face)
    # logger.debug("Dofs per edge:\n"+f"{dofs_per_edge=}")

    # ### Definition of Dirichlet BCs
    # This also depends on number of modes and can only be defined after
    # distribution of dofs
    origin = dofmap.grid.locate_entities_boundary(0, point_at([0.0, 0.0, 0.0]))
    bottom_right = dofmap.grid.locate_entities_boundary(0, point_at([10.0, 0.0, 0.0]))
    bc_dofs = []
    for vertex in origin:
        bc_dofs += dofmap.entity_dofs(0, vertex)
    for vertex in bottom_right:
        dofs = dofmap.entity_dofs(0, vertex)
        bc_dofs.append(dofs[1])  # constrain uy, but not ux
    assert len(bc_dofs) == 3
    bc_dofs = np.array(bc_dofs)

    lhs = defaultdict(list)
    rhs = defaultdict(list)
    bc_mat = defaultdict(list)
    local_bases = []

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)

        # select active modes
        local_basis = select_modes(bases[ci], num_max_modes[ci], dofs_per_edge[ci])
        local_bases.append(local_basis)
        B = A.source.from_numpy(local_basis)  # type: ignore
        A_local = project(A, B, B)
        b_local = project(b, B, None)
        element_matrix = A_local.matrix  # type: ignore
        element_vector = b_local.matrix  # type: ignore

        for l, x in enumerate(dofs):
            if x in bc_dofs:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(0.0)
            else:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(element_vector[l, 0])

            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:
                    # Note: in the MOR context set diagonal to zero
                    # for the matrices arising from a_q
                    if x == y:
                        if x not in lhs["diagonals"]:  # only set diagonal entry once
                            lhs["rows"].append(x)
                            lhs["cols"].append(y)
                            lhs["data"].append(0.0)
                            lhs["diagonals"].append(x)
                            bc_mat["rows"].append(x)
                            bc_mat["cols"].append(y)
                            bc_mat["data"].append(1.0)
                            bc_mat["diagonals"].append(x)
                else:
                    lhs["rows"].append(x)
                    lhs["cols"].append(y)
                    lhs["data"].append(element_matrix[l, k])

        lhs["indexptr"].append(len(lhs["rows"]))
        rhs["indexptr"].append(len(rhs["rows"]))

    Ndofs = dofmap.num_dofs
    data = np.array(lhs["data"])
    rows = np.array(lhs["rows"])
    cols = np.array(lhs["cols"])
    indexptr = np.array(lhs["indexptr"])
    shape = (Ndofs, Ndofs)
    options = None
    op = COOMatrixOperator(
        (data, rows, cols),
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=parameters,
        solver_options=options,
        name="K",
    )

    # ### Add matrix to account for BCs
    bc_array = coo_array(
        (bc_mat["data"], (bc_mat["rows"], bc_mat["cols"])), shape=shape
    )
    bc_array.eliminate_zeros()
    bc_op = NumpyMatrixOperator(
        bc_array.tocsr(), op.source.id, op.range.id, op.solver_options, "bc_mat"
    )

    lincomb = LincombOperator([op, bc_op], [1.0, 1.0])

    data = np.array(rhs["data"])
    rows = np.array(rhs["rows"])
    cols = np.array(rhs["cols"])
    indexptr = np.array(rhs["indexptr"])
    shape = (Ndofs, 1)
    rhs_op = COOMatrixOperator(
        (data, rows, cols),
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=Parameters({}),
        solver_options=options,
        name="F",
    )
    return lincomb, rhs_op, local_bases


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem
    from .fom import discretize_fom

    # ### logger
    set_defaults(
        {"pymor.core.logger.getLogger.filename": beam.log_run_locrom(args.distr, args.name)}
    )
    logger = getLogger(Path(__file__).stem, level="DEBUG")

    # ### Discretize FOM
    fom = discretize_fom(beam)
    h1_product = fom.products["h1_0_semi"]

    # ### Discretize operators on subdomain
    gdim = beam.gdim
    unit_cell_domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, unit_cell_domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()
    top_tag = int(137)
    omega.create_facet_tags({"top": top_tag})

    # FE space
    degree = beam.fe_deg
    fe = element("P", omega.grid.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)

    # base material
    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # Problem on unit cell domain
    problem = LinElaSubProblem(omega, V, phases=mat)
    loading = fem.Constant(
        omega.grid, (default_scalar_type(0.0), default_scalar_type(-10.0))
    )
    problem.add_neumann_bc(top_tag, loading)

    # Full operators
    problem.setup_solver()
    problem.assemble_matrix()
    problem.assemble_vector()
    A = FenicsxMatrixOperator(problem.A, V, V)
    b = VectorOperator(source.make_array([problem.b]))  # type: ignore

    # ### Multiscale Problem
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    coarse_grid = beam_problem.coarse_grid
    dofmap = DofMap(coarse_grid)

    # ### Reduced bases
    bases_folder = beam.bases_path(args.distr, args.name)
    num_cells = beam.nx * beam.ny
    bases_loader = BasesLoader(bases_folder, num_cells)
    bases, num_max_modes = bases_loader.read_bases()
    num_max_modes_per_cell = np.amax(num_max_modes, axis=1)
    num_min_modes_per_cell = np.amin(num_max_modes, axis=1)
    max_modes = np.amax(num_max_modes_per_cell)
    min_modes = np.amin(num_min_modes_per_cell)
    logger.info(f"Global minimum number of modes per edge is: {min_modes}.")
    logger.info(f"Global maximum number of modes per edge is: {max_modes}.")

    # ### ROM Assembly and Error Analysis
    P = fom.parameters.space(beam.mu_range)
    with new_rng(beam.validation_seed):
        validation_set = P.sample_randomly(args.num_test)

    # better not create functions inside loops
    u_rb = fem.Function(fom.solution_space.V)
    u_loc = fem.Function(V)

    max_errors = []
    max_relerrors = []

    # TODO set appropriate value for number of modes
    num_fine_scale_modes = list(range(0, max_modes + 1, 2))

    for nmodes in num_fine_scale_modes:
        operator, rhs, local_bases = assemble_system(
            nmodes, dofmap, A, b, bases, num_max_modes, fom.parameters
        )
        rom = StationaryModel(operator, rhs, name="locROM")

        fom_solutions = fom.solution_space.empty()
        rom_solutions = fom.solution_space.empty()

        err_norms = []
        for mu in validation_set:
            U_fom = fom.solve(mu)  # is this cached or computed everytime?
            fom_solutions.append(U_fom)
            U_rb_ = rom.solve(mu)

            reconstruct(U_rb_.to_numpy(), dofmap, local_bases, u_loc, u_rb)
            # copy seems necessary here
            # without it I get a PETSC ERROR (segmentation fault)
            U_rom = fom.solution_space.make_array([u_rb.vector.copy()])  # type: ignore
            rom_solutions.append(U_rom)

        err = fom_solutions - rom_solutions
        fom_norms = fom_solutions.norm(h1_product)
        err_norms = err.norm(h1_product)
        max_err = np.max(err_norms)
        logger.debug(f"{nmodes=}\tnum_dofs: {dofmap.num_dofs}\t{max_err=}")
        max_errors.append(max_err)
        max_relerrors.append(max_err / fom_norms[np.argmax(err_norms)])

    if args.output is not None:
        np.savetxt(args.output, np.vstack((num_fine_scale_modes, max_relerrors)).T, delimiter=",", header="modes, error")

    if args.show:
        import matplotlib.pyplot as plt

        plt.title("ROM error relative to FOM")
        plt.semilogy(num_fine_scale_modes, max_relerrors, "k-o")
        plt.ylabel("Rel. error")
        plt.xlabel("Number of fine scale basis functions per edge")
        plt.show()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve the beam problem with the localized ROM and compute the error relative to the FOM solution for a test set (sampled randomly)."
    )
    parser.add_argument(
        "distr",
        type=str,
        help="The distribution used for sampling.",
        choices=("normal", "multivariate_normal"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument(
        "num_test", type=int, help="Size of the test set used for validation."
    )
    parser.add_argument("--show", action="store_true", help="show error plot.")
    parser.add_argument("--output", type=str, help="Path (.csv) to write relative error.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
