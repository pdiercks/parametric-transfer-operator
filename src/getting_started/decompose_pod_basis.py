from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmshio
from basix.ufl import element
import numpy as np

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer

from multi.debug import plot_modes
from multi.bcs import BoundaryDataFactory
from multi.extension import extend
from multi.misc import locate_dofs, x_dofs_vectorspace
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.basis_construction import compute_phi


def main(args):
    """decompose pod basis into coarse and fine scale parts"""
    from .tasks import beam
    gdim = 2
    domain, _, _ = gmshio.read_from_msh(beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim)
    omega = RectangularSubdomain(12, domain)
    # create grids for later use
    omega.create_coarse_grid()
    omega.create_edge_grids()

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
    problem.setup_coarse_space()
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    # ### Coarse scale basis
    nodes = omega.coarse_grid.geometry.x
    phi_vectors = compute_phi(problem, nodes)
    phi = source.make_array(phi_vectors)

    # ### Full POD basis
    pod_basis_data = np.load(beam.loc_pod_modes(args.distribution))
    pod_basis = source.from_numpy(pod_basis_data)

    # ### subtract coarse scale part
    xdofs = x_dofs_vectorspace(V)
    node_dofs = locate_dofs(xdofs, nodes)
    coarse_dofs = pod_basis.dofs(node_dofs)
    u_coarse = phi.lincomb(coarse_dofs)
    u_fine = pod_basis - u_coarse
    zero_dofs = u_fine.dofs(node_dofs)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # from multi.debug import plot_modes
    # bottom_modes = u_fine.dofs(problem.V_to_L.get('bottom'))
    # L = problem.edge_spaces['fine'].get('bottom')
    # mask = np.s_[:5]
    # plot_modes(L, 'b', bottom_modes, "x", mask)

    # ### Restriction of fine scale part to edges
    bc_factory = BoundaryDataFactory(problem.domain.grid, problem.V)
    # edge_modes = dict()
    edges = set(["left", "bottom", "right", "top"])
    zero_function = fem.Function(problem.V)
    zero_function.x.array[:] = 0.
    boundary_data = list()

    for edge, dofs in problem.V_to_L.items():
        modes = u_fine.dofs(dofs)
        # create BCs for extension of each mode
        zero_boundaries = list(edges.difference([edge]))
        for mode in modes:
            bc = []
            g = bc_factory.create_function_values(mode, dofs)
            bc.append({"value": g, "boundary": problem.domain.str_to_marker(edge),
                       "method": "geometrical"})
            for boundary in zero_boundaries:
                bc.append({"value": zero_function, "boundary": problem.domain.str_to_marker(boundary),
                           "method": "geometrical"})
            assert len(bc) == 4
            boundary_data.append(bc)
        # edge_modes[edge] = u_fine.dofs(dofs)

    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            }
    extensions = extend(problem, boundary_data=boundary_data, petsc_options=petsc_options)
    viz = FenicsxVisualizer(source)
    U = source.make_array(extensions)
    viz.visualize(U, filename="./ext.xdmf")

    breakpoint()
    # TODO do again compression over edge sets | not good
    # TODO extend final edge functions
    # TODO write coarse and fine scale basis


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("distribution", type=str, help="The distribution used in the range approximation.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
