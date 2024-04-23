
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
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from multi.projection import relative_error, absolute_error

TY = -5.0e-2
EMOD = 1.0
POISSON = 0.3


def compute_reference_solution(mshfile, degree):
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

    top_locator = plane_at(1.0, "y")
    bottom_locator = plane_at(0.0, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    material = LinearElasticMaterial(gdim=2, E=EMOD, NU=POISSON)
    V = fem.functionspace(domain, ("P", degree, (2,)))
    problem = LinearElasticityProblem(omega, V, phases=material)
    zero = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(0.0)))
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )
    problem.add_dirichlet_bc(value=zero, boundary=bottom_locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    u = problem.solve()
    return u


def compute_other(domain, V):
    top_locator = plane_at(1.0, "y")
    bottom_locator = plane_at(0.0, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    zero = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(0.0)))
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )

    class MyModel(LinearProblem):
        def __init__(self, domain, V, E, NU):
            super().__init__(domain, V)
            self.mat = LinearElasticMaterial(gdim=2, E=E, NU=NU)

        @property
        def form_lhs(self):
            i, j, k, l = ufl.indices(4)
            gdim = self.domain.gdim
            Id = ufl.Identity(gdim)

            lame_1 = self.mat.lambda_1
            lame_2 = self.mat.lambda_2

            tetrad_ijkl = lame_1 * Id[i, j] * Id[k, l] + lame_2 * (
                    Id[i, k] * Id[j, l] + Id[i, l] * Id[j, k] # type: ignore
            )
            grad_u = self.trial[k].dx(l) # type: ignore
            grad_v = self.test[i].dx(j)  # type: ignore
            return grad_v * tetrad_ijkl * grad_u * ufl.dx

        @property
        def form_rhs(self):
            v = self.test
            rhs = ufl.inner(zero, v) * ufl.dx

            if self._bc_handler.has_neumann:
                rhs += self._bc_handler.neumann_bcs

            return rhs

    problem = MyModel(omega, V, E=EMOD, NU=POISSON)
    problem.add_dirichlet_bc(value=zero, boundary=bottom_locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    u = problem.solve()
    return u


def main():
    from .tasks import example

    # Generate physical subdomain
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg
    uref = compute_reference_solution(parent_subdomain_msh, degree)
    V = uref.function_space
    u = compute_other(V.mesh, V)

    inner_product = InnerProduct(V, "mass")
    prod_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(prod_mat, V, V)
    source = FenicsxVectorSpace(V)

    REF = source.make_array([uref.vector]) # type: ignore
    U = source.make_array([u.vector]) # type: ignore
    abs_err = absolute_error(REF, U, product)
    rel_err = relative_error(REF, U, product)
    print(f"{abs_err=}")
    print(f"{rel_err=}")


if __name__ == "__main__":
    main()
