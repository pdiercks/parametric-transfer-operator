import typing
import ufl
from dolfinx import fem, default_scalar_type

from multi.domain import Domain
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial


class ParaGeomLinEla(LinearProblem):
    """Represents a geometrically parametrized linear elastic problem."""

    def __init__(self, domain: Domain, V: fem.FunctionSpace, E: typing.Union[float, fem.Constant], NU: typing.Union[float, fem.Constant], d: fem.Function):
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
        self.dx = ufl.Measure('dx', domain=domain.grid)

    def weighted_stress(self, w: ufl.TrialFunction): # type: ignore
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
                Id[i, m] * Id[k, l] + Id[i, l] * Id[k, m] # type: ignore
        )
        grad_u_ml = w[m].dx(p) * Finv[p, l] # type: ignore
        sigma_ij = ufl.as_tensor(tetrad_ikml * grad_u_ml * FinvT[k, j] * detF, (i, j)) # type: ignore
        return sigma_ij

    @property
    def form_lhs(self):
        grad_v_ij = ufl.grad(self.test)
        sigma_ij = self.weighted_stress(self.trial)
        return ufl.inner(grad_v_ij, sigma_ij) * self.dx

    @property
    def form_rhs(self):
        v = self.test
        zero = fem.Constant(self.domain.grid, (default_scalar_type(0.0), default_scalar_type(0.0)))
        rhs = ufl.inner(zero, v) * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs
