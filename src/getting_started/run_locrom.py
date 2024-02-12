from pathlib import Path
from collections import defaultdict

from scipy.sparse import coo_array
import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from pymor.parameters.base import Parameters
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.algorithms.projection import project
from pymor.models.basic import StationaryModel

from multi.boundary import point_at
from multi.domain import RectangularSubdomain
from multi.dofmap import DofMap
from multi.io import BasesLoader, select_modes
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem


def reconstruct(U_rb: np.ndarray, dofmap: DofMap, bases: list[np.ndarray], subdomains: list[np.ndarray], V: fem.FunctionSpaceBase) -> fem.Function:
    """Reconstructs rom solution on the global domain.

    Args:
        Urb: ROM solution in the reduced space.
        dofmap: The dofmap of the reduced space.
        bases: Local basis for each subdomain.
        subdomains: The cells of each subdomain.
        V: The global FE space (``fom.solution_space``).

    Returns:
        u: The reconstructed solution.

    """
    domain = V.mesh
    u = fem.Function(V)
    u_view = u.x.array # type: ignore
    fe = V.ufl_element()

    for i_subdomain, cells in enumerate(subdomains):

        submesh, cell_map, _, _ = mesh.create_submesh(domain, domain.topology.dim, cells)
        V_sub = fem.FunctionSpace(submesh, fe)
        u_local = fem.Function(V_sub)
        u_local_view = u_local.x.array # type: ignore

        # fill u_local with rom solution
        basis = bases[i_subdomain]
        dofs = dofmap.cell_dofs(i_subdomain)
        u_local_view = U_rb[0, dofs] @ basis

        # fill global u
        num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
        for cell in range(num_sub_cells):
            child_dofs = V_sub.dofmap.cell_dofs(cell)
            parent_dofs = V.dofmap.cell_dofs(cell_map[cell])
            for parent, child in zip(parent_dofs, child_dofs):
                for b in range(V_sub.dofmap.bs):
                    u_view[parent*V.dofmap.bs+b] = u_local_view[child*V_sub.dofmap.bs+b]
    return u # type: ignore


def assemble_system(logger, num_modes: int, dofmap: DofMap, A: FenicsxMatrixOperator, b: VectorOperator, bases: list[np.ndarray], num_max_modes: np.ndarray, parameters: Parameters):
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
        bc_dofs.append(dofs[1]) # constrain uy, but not ux
    assert len(bc_dofs) == 3
    bc_dofs = np.array(bc_dofs)

    lhs = defaultdict(list)
    rhs = defaultdict(list)
    bc_mat = defaultdict(list)
    local_bases = []

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)

        # select active modes
        stuff = select_modes(
                bases[ci], num_max_modes[ci], dofs_per_edge[ci]
                )
        local_basis = stuff.copy()
        local_bases.append(local_basis)
        B = A.source.from_numpy(local_basis) # type: ignore
        A_local = project(A, B, B)
        b_local = project(b, B, None)
        element_matrix = A_local.matrix # type: ignore
        element_vector = b_local.matrix # type: ignore

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
                        if x not in lhs["diagonals"]: # only set diagonal entry once
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
    op = COOMatrixOperator((data, rows, cols), indexptr, dofmap.num_cells, shape, parameters=parameters, solver_options=options, name="K")

    # ### Add matrix to account for BCs
    bc_array = coo_array((bc_mat["data"], (bc_mat["rows"], bc_mat["cols"])), shape=shape)
    bc_array.eliminate_zeros()
    bc_op = NumpyMatrixOperator(bc_array.tocsr(), op.source.id, op.range.id, op.solver_options, "bc_mat")

    lincomb = LincombOperator([op, bc_op], [1., 1.])

    data = np.array(rhs["data"])
    rows = np.array(rhs["rows"])
    cols = np.array(rhs["cols"])
    indexptr = np.array(rhs["indexptr"])
    shape = (Ndofs, 1)
    rhs_op = COOMatrixOperator((data, rows, cols), indexptr, dofmap.num_cells, shape, parameters={}, solver_options=options, name="F")
    return lincomb, rhs_op, local_bases


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem
    from .fom import discretize_fom

    # ### logger
    set_defaults(
        {"pymor.core.logger.getLogger.filename": beam.log_run_locrom(args.distr)}
    )
    logger = getLogger(Path(__file__).stem, level="DEBUG")

    # ### Discretize FOM

    # read cell tags for reconstruction of reduced solution
    with XDMFFile(MPI.COMM_WORLD, beam.fine_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, "subdomains")

    fom = discretize_fom(beam)
    h1_product = fom.products["h1_0_semi"]

    # ### Discretize operators on subdomain
    gdim = beam.gdim
    domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()
    top_tag = int(137)
    omega.create_facet_tags({"top": top_tag})

    # FE space
    degree = beam.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)

    # base material
    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # Problem on unit cell domain
    problem = LinElaSubProblem(omega, V, phases=(mat,))
    loading = fem.Constant(
            omega.grid, (default_scalar_type(0.0), default_scalar_type(-10.0)))
    problem.add_neumann_bc(top_tag, loading)

    # Full operators
    problem.setup_solver()
    problem.assemble_matrix()
    problem.assemble_vector()
    A = FenicsxMatrixOperator(problem.A, V, V)
    b = VectorOperator(source.make_array([problem.b])) # type: ignore

    # ### Multiscale Problem
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    coarse_grid = beam_problem.coarse_grid
    dofmap = DofMap(coarse_grid)

    # ### Reduced bases
    bases_folder = beam.bases_path(args.distr)
    num_cells = beam.nx * beam.ny
    bases_loader = BasesLoader(bases_folder, num_cells)
    bases, num_max_modes = bases_loader.read_bases()
    num_max_modes_per_cell = np.amax(num_max_modes, axis=1)
    num_min_modes_per_cell = np.amin(num_max_modes, axis=1)
    max_modes = np.amax(num_max_modes_per_cell)
    min_modes = np.amin(num_min_modes_per_cell)
    logger.info(f"Global maximum number of modes per edge is: {max_modes}.")

    # ### Cell set for each subdomain
    subdomains = []
    for j in range(1, 11):
        subdomains.append(cell_tags.find(j))

    P = fom.parameters.space(beam.mu_range)
    validation_set = P.sample_randomly(1)
    # num_fine_scale_modes = list(range(0,66+1,1))
    num_fine_scale_modes = list(range(min_modes))

    vvv = FenicsxVisualizer(fom.solution_space)

    # ### ROM Assembly and Error Analysis
    errors = []
    for mu in validation_set:
        U_fom = fom.solve(mu)
        # vvv.visualize(U_fom, filename="./work/debug_fom.bp")

        for nmodes in num_fine_scale_modes:

            operator, rhs, local_bases = assemble_system(logger, nmodes, dofmap, A, b, bases, num_max_modes, fom.parameters)
            # bc_dofs = operator.operators[-1].matrix.indices

            NDOFs = dofmap.num_dofs

            # FIXME rom.solve(mu) is not working as expected!
            # Urb = rom.solve(mu)

            # FIXME
            # StationaryModel does some caching for evaluation for some mu
            # rom = StationaryModel(operator, rhs, name=f"rom_{dofmap.num_dofs}")
            # data = rom.compute(solution=True, mu=mu)
            # Urb = data['solution']

            fixed_op = operator.assemble(mu)
            fixed_rhs = rhs.as_range_array()
            K = fixed_op.matrix.todense()
            F = fixed_rhs.to_numpy().flatten()
            # κ = np.linalg.cond(K)
            # logger.debug(f"Condition number {κ=}")
            Urb = np.linalg.solve(K, F)

            u_rb = reconstruct(Urb.reshape(1, Urb.size), dofmap, local_bases, subdomains, fom.solution_space.V) # type: ignore
            # u_rb = reconstruct(Urb.to_numpy(), dofmap, local_bases, subdomains, fom.solution_space.V) # type: ignore
            U_rom = fom.solution_space.make_array([u_rb.vector])
            # vvv.visualize(U_rom, filename="./work/debug_rom.bp")

            ERR = U_fom - U_rom
            ufom_norm = U_fom.norm(h1_product)
            urom_norm = U_rom.norm(h1_product)
            if nmodes == 28:
                vvv.visualize(ERR, filename="./work/debug_err.bp")

            err = ERR.norm(h1_product)
            errors.append(err[0])
            logger.debug(f"{nmodes=}\t{NDOFs=}\terror: {err}")
            logger.debug(f"{ufom_norm=}")
            logger.debug(f"{urom_norm=}")

            # FIXME
            # with nmodes>=29 the error starts to increase
            # what is going wrong?

    import matplotlib.pyplot as plt
    plt.semilogy(np.arange(len(errors)), errors, "k-o")
    plt.show()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Solve the beam problem with the localized ROM."
    )
    parser.add_argument(
        "distr",
        type=str,
        help="The distribution used for sampling.",
        choices=("normal", "multivariate_normal"),
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
