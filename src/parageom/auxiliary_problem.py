from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem


def discretize_auxiliary_problem(degree, mu):
    # TODO read reference domain
    omega = RectangularDomain(domain)
    emod = fem.Constant(omega.grid, default_scalar_type(1.0))
    nu = fem.Constant(omega.grid, default_scalar_type(0.25))
    gdim = omega.grid.geometry.dim
    mat = LinearElasticMaterial(gdim, E=emod, NU=nu)
    ve = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(domain, ve)
    problem = LinearElasticityProblem(omega, V, phases=mat)

    # ### Dirichlet bcs
    # fix displacement for bottom, left, right and top boundary
    u_zero = fem.Constant(omega.grid, (default_scalar_type(0.0), ) * gdim)
    for boundary in omega.boundaries:
        marker = omega.str_to_marker(boundary)
        problem.add_dirichlet_bc(u_zero, boundary=marker, method="geometrical")

    # similar to extension problem:
    # - prepare A with all bcs applied
    # - create solver
    # - for each solve: create new rhs vector

    # maybe use a factory to create boundary data g(μ)
    # such that g(μ)=0 on δΩ 
    # and       g(μ)=map(μ) on δΩ_void
    # map(μ) can be a numpy function
    # need to determine dofs for both boundaries
    # and then set the values on the g.x.array directly

    # maybe wrap everything as problem.solve(mu) for convenience

    # maybe check, but I guess I cannot express this as pymor model


def main():
    # define training set
    # parameters = {"R": 1}

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    degree = 1

    # discretize auxiliary problem for μ
    problem = discretize_auxiliary_problem(degree, mu)
    pass


if __name__ == "__main__":
    main()
