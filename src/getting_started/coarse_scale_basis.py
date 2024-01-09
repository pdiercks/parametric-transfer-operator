from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmshio
from basix.ufl import element
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.basis_construction import compute_phi


def main():
    """compute the coarse scale basis via extension of bilinear shape functions"""
    from .tasks import beam
    gdim = 2
    domain, _, _ = gmshio.read_from_msh(beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim)
    omega = RectangularSubdomain(12, domain)

    # ### FE spaces
    degree = beam.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)

    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # ### Problem on unit cell domain
    omega.create_coarse_grid()
    nodes = omega.coarse_grid.geometry.x
    problem = LinElaSubProblem(omega, V, phases=(mat,))
    phi = compute_phi(problem, nodes)
    breakpoint()

    # TODO load POD basis
    # TODO subtract phi
    # TODO restrict to edges
    # TODO do again compression over edge sets | not good
    # TODO extend final edge functions
    # TODO write coarse and fine scale basis


if __name__ == "__main__":
    main()
