from pathlib import Path
import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.constructions import VectorOperator
from pymor.algorithms.projection import project

from multi.boundary import point_at
from multi.domain import RectangularSubdomain
from multi.dofmap import DofMap
from multi.io import BasesLoader, select_modes
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem


# rename to assemble_system
# use defaultdict
# assemble lhs and rhs in the same loop
def assemble_operator(num_modes: int, dofmap: DofMap, A: FenicsxMatrixOperator, bases: list[np.ndarray], num_max_modes: np.ndarray, bc_dofs: np.ndarray):
    from .locmor import COOMatrixOperator

    dofs_per_vertex = 2
    dofs_per_face = 0

    dofs_per_edge = num_max_modes.copy()
    dofs_per_edge[num_max_modes > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vertex, dofs_per_edge, dofs_per_face)

    diagonals = []
    data = []
    rows = []
    cols = []
    indexptr = []

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)

        # select active modes
        local_basis = select_modes(
                bases[ci], num_max_modes[ci], dofs_per_edge[ci]
                )
        B = A.source.from_numpy(local_basis)
        A_local = project(A, B, B)
        element_matrix = A_local.matrix

        for l, x in enumerate(dofs):
            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:
                    if x == y:
                        if x not in diagonals:
                            rows.append(x)
                            cols.append(y)
                            data.append(0.0)
                            diagonals.append(x)
                else:
                    rows.append(x)
                    cols.append(y)
                    data.append(element_matrix[l, k])
        indexptr.append(len(rows))
    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)
    indexptr = np.array(indexptr)
    N = dofmap.num_dofs
    shape = (N, N)
    options = None
    op = COOMatrixOperator((data, rows, cols), indexptr, dofmap.num_cells, shape, solver_options=options, name="K")
    return op


def assemble_rhs():
    # assemble rhs using COOMatrixOperator works
    # however, cannot pass COOMatrixOperator as rhs
    # to StationaryModel.
    # constraints:
    # assert rhs.range == operator.range and rhs.source.is_scalar and rhs.linear
    # How to resolve this?
    print("...")


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem

    # ### logger
    set_defaults(
        {"pymor.core.logger.getLogger.filename": beam.log_run_locrom(args.distr)}
    )
    logger = getLogger(Path(__file__).stem, level="INFO")

    gdim = beam.gdim
    domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()
    top_tag = int(137)
    omega.create_facet_tags({"top": top_tag})

    # ### FE spaces
    degree = beam.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)

    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # ### Problem on unit cell domain
    problem = LinElaSubProblem(omega, V, phases=(mat,))
    loading = fem.Constant(
            omega.grid, (default_scalar_type(0.0), default_scalar_type(-10.0)))
    problem.add_neumann_bc(top_tag, loading)

    # ### Full operators
    problem.setup_solver()
    problem.assemble_matrix()
    problem.assemble_vector()
    A = FenicsxMatrixOperator(problem.A, V, V)
    b = VectorOperator(source.make_array([problem.b]))
    breakpoint()

    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    coarse_grid = beam_problem.coarse_grid
    dofmap = DofMap(coarse_grid)

    # ### Definition of Dirichlet BCs
    origin = coarse_grid.locate_entities_boundary(0, point_at([0.0, 0.0, 0.0]))
    bottom_right = coarse_grid.locate_entities_boundary(0, point_at([10.0, 0.0, 0.0]))
    bc_dofs = []
    for vertex in origin:
        bc_dofs += dofmap.entity_dofs(0, vertex)
    for vertex in bottom_right:
        dofs = dofmap.entity_dofs(0, vertex)
        bc_dofs.append(dofs[1]) # constrain uy, but not ux
    assert len(bc_dofs) == 3
    bc_dofs = np.array(bc_dofs)
    bc_vals = np.zeros_like(bc_dofs)

    # implement def assemble_rom
    # assemble rom for different number of modes
    # for each rom: compute error over validation set
    bases_folder = beam.bases_path(args.distr)
    num_cells = beam.nx * beam.ny
    bases_loader = BasesLoader(bases_folder, num_cells)
    bases, num_max_modes = bases_loader.read_bases()
    num_max_modes_per_cell = np.amax(num_max_modes, axis=1)
    max_modes = np.amax(num_max_modes_per_cell)
    logger.info(f"Global maximum number of modes per edge is: {max_modes}.")

    breakpoint()
    num_fine_scale_modes = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    for nmodes in num_fine_scale_modes:


        K = assemble_operator(nmodes, dofmap, A, bases, num_max_modes, bc_dofs)
        # TODO assemble RHS as COOMatrixOperator?

        # initialize Stationary model with reduced operators
        # final lhs should be LincombOperator([COOMatrixOperator, BCsOperator], [1, 1])


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
