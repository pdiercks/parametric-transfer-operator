from pathlib import Path
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from basix.ufl import element
import numpy as np

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.bindings.fenicsx import FenicsxVectorSpace

from multi.misc import locate_dofs, x_dofs_vectorspace
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.basis_construction import compute_phi


def main(args):
    """decompose pod basis into coarse and fine scale parts"""
    from .tasks import beam
    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_decompose_pod_basis(
                args.distribution, args.configuration
            ),
        }
    )
    logger = getLogger(Path(__file__).stem, level="INFO")

    gdim = beam.gdim
    domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()

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

    # ### Full POD basis
    pod_basis_data = np.load(beam.loc_pod_modes(args.distribution, args.configuration))
    pod_basis = source.from_numpy(pod_basis_data)
    logger.info(f"Size of POD basis: {len(pod_basis)}.")

    # ### Coarse scale basis
    vertices = omega.coarse_grid.topology.connectivity(2, 0).links(0)
    x_vertices = mesh.compute_midpoints(omega.coarse_grid, 0, vertices)
    phi_vectors = compute_phi(problem, x_vertices)
    phi = source.make_array(phi_vectors)

    # ### subtract coarse scale part
    xdofs = x_dofs_vectorspace(V)
    node_dofs = locate_dofs(xdofs, x_vertices)
    coarse_dofs = pod_basis.dofs(node_dofs)
    u_coarse = phi.lincomb(coarse_dofs)
    u_fine = pod_basis - u_coarse
    zero_dofs = u_fine.dofs(node_dofs)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # ### Restriction of fine scale part to edges
    fine_scale_modes = {}
    for edge, dofs in problem.V_to_L.items():
        fine_scale_modes[edge] = u_fine.dofs(dofs)

    np.savez(
        beam.fine_scale_edge_modes_npz(args.distribution, args.configuration).as_posix(),
        **fine_scale_modes,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "distribution",
        type=str,
        help="The distribution used in the range approximation.",
    )
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
