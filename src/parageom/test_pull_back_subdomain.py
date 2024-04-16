import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio, XDMFFile
import ufl

from multi.boundary import plane_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinearProblem


def compute_reference_solution(domain, degree):
    top_locator = plane_at(1.0, "y")
    bottom_locator = plane_at(0.0, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, marked = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    material = LinearElasticMaterial(gdim=2, E=20e3, NU=0.3)
    V = fem.functionspace(domain, ("P", degree, (2,)))
    problem = LinearElasticityProblem(omega, V, phases=material)
    zero = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(0.0)))
    traction = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(-1000.)))
    problem.add_dirichlet_bc(value=zero, boundary=bottom_locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    u = problem.solve()
    return u


def compute_pull_back_model_solution(domain, trafo_disp):
    top_locator = plane_at(1.0, "y")
    bottom_locator = plane_at(0.0, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, marked = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    zero = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(0.0)))
    traction = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(-1000.)))
    V = trafo_disp.function_space
    
    class MyModel(LinearProblem):
        def __init__(self, domain, V, E, NU, d):
            super().__init__(domain, V)
            self.mat = LinearElasticMaterial(gdim=2, E=E, NU=NU)
            self.d = d

        @property
        def form_lhs(self):
            i, j, k, l, m = ufl.indices(5)
            gdim = self.domain.gdim
            Id = ufl.Identity(gdim)

            # pull back stuff
            F = Id + ufl.grad(self.d)
            detF = ufl.det(F)
            C = F.T * F
            inv_C = ufl.inv(C)

            lame_1 = self.mat.lambda_1
            lame_2 = self.mat.lambda_2

            tetrad_ijkl = lame_1 * Id[i, j] * Id[k, l] + lame_2 * (
                    Id[i, k] * Id[j, l] + Id[i, l] * Id[j, k])
            grad_u = self.trial[k].dx(m) * inv_C[m, l] * detF
            grad_v = self.test[i].dx(j)
            return grad_v * tetrad_ijkl * grad_u * ufl.dx

        @property
        def form_rhs(self):
            v = self.test
            rhs = ufl.inner(zero, v) * ufl.dx

            if self._bc_handler.has_neumann:
                rhs += self._bc_handler.neumann_bcs

            return rhs

    problem = MyModel(omega, V, E=20e3, NU=0.3, d=trafo_disp)
    problem.add_dirichlet_bc(value=zero, boundary=bottom_locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    u = problem.solve()
    return u

def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # Generate physical subdomain
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    # FIXME
    # limited to degree 1
    # ValueError in line 117 in auxiliary_problem.py if degree == 2
    degree = example.geom_deg
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, example.parameters["subdomain"]
    )
    mu = aux.parameters.parse([0.127])
    d = fem.Function(aux.problem.V)
    aux.solve(d, mu)  # type: ignore

    parent_subdomain = gmshio.read_from_msh(
        parent_subdomain_msh, MPI.COMM_WORLD, gdim=2
    )[0]

    u = compute_pull_back_model_solution(aux.problem.domain.grid, d)

    with XDMFFile(parent_subdomain.comm, "pully.xdmf", "w") as xdmf:
        xdmf.write_mesh(parent_subdomain)
        xdmf.write_function(u, t=0.0)

    x_subdomain = parent_subdomain.geometry.x
    disp = np.pad(
        d.x.array.reshape(x_subdomain.shape[0], -1),  # type: ignore
        pad_width=[(0, 0), (0, 1)],
    )
    x_subdomain += disp
    u_ref = compute_reference_solution(parent_subdomain, degree)

    with XDMFFile(parent_subdomain.comm, "reference.xdmf", "w") as xdmf:
        xdmf.write_mesh(parent_subdomain)
        xdmf.write_function(u_ref, t=0.0)

    # compare
    err = u.x.array - u_ref.x.array
    print("error in euclidean norm: ", np.linalg.norm(err))
    # np.linalg.norm(err) ~ 5% if degree=1
    # np.linalg.norm(err) ~ 9% if degree=2
    # Not sure if this is expected or some mistake.
    # ASAIK the pull back should be exact.


if __name__ == "__main__":
    main()
