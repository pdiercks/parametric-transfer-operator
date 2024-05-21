import typing
import ufl

import dolfinx as df

from multi.domain import Domain, RectangularDomain
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial


class ParaGeomLinEla(LinearProblem):
    """Represents a geometrically parametrized linear elastic problem."""

    def __init__(
        self,
        domain: Domain,
        V: df.fem.FunctionSpace,
        E: typing.Union[float, df.fem.Constant],
        NU: typing.Union[float, df.fem.Constant],
        d: df.fem.Function,
    ):
        """Initialize linear elastic model with pull back.

        Args:
            domain: The parent domain.
            V: FE space.
            E: Young's modulus.
            NU: Poisson ratio.
            d: parametric transformation displacement field.
        """
        super().__init__(domain, V)
        self.mat = LinearElasticMaterial(gdim=domain.gdim, E=E, NU=NU)
        self.d = d
        self.dx = ufl.Measure("dx", domain=domain.grid)

    def weighted_stress(self, w: ufl.TrialFunction):  # type: ignore
        """Returns weighted stress as UFL form.

        Args:
            w: TrialFunction.

        Note:
            The weighted stress depends on the current value of the
            transformation displacement field.

        """
        lame_1 = self.mat.lambda_1
        lame_2 = self.mat.lambda_2

        gdim = self.domain.gdim
        Id = ufl.Identity(gdim)

        # pull back
        F = Id + ufl.grad(self.d)  # type: ignore
        detF = ufl.det(F)
        Finv = ufl.inv(F)
        FinvT = ufl.inv(F.T)

        i, j, k, l, m, p = ufl.indices(6)
        tetrad_ikml = lame_1 * Id[i, k] * Id[m, l] + lame_2 * (
            Id[i, m] * Id[k, l] + Id[i, l] * Id[k, m]  # type: ignore
        )
        grad_u_ml = w[m].dx(p) * Finv[p, l]  # type: ignore
        sigma_ij = ufl.as_tensor(tetrad_ikml * grad_u_ml * FinvT[k, j] * detF, (i, j))  # type: ignore
        return sigma_ij

    @property
    def form_lhs(self):
        grad_v_ij = ufl.grad(self.test)
        sigma_ij = self.weighted_stress(self.trial)
        return ufl.inner(grad_v_ij, sigma_ij) * self.dx

    @property
    def form_rhs(self):
        v = self.test
        zero = df.fem.Constant(
            self.domain.grid, (df.default_scalar_type(0.0), df.default_scalar_type(0.0))
        )
        rhs = ufl.inner(zero, v) * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs


def discretize_subdomain_operator(example):
    from .auxiliary_problem import discretize_auxiliary_problem
    from .matrix_based_operator import FenicsxMatrixBasedOperator

    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, ftags, example.parameters["subdomain"]
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    EMOD = 1. # dimless formulation
    POISSON = example.poisson_ratio
    domain = aux.problem.domain.grid
    omega = RectangularDomain(domain)
    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d)

    # ### wrap as pymor model
    def param_setter(mu):
        d.x.array[:] = 0.0
        aux.solve(d, mu)
        d.x.scatter_forward()

    params = {"R": 1}
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs, params, param_setter=param_setter, name="ParaGeom"
    )
    return operator
