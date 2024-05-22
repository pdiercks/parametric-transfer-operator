import typing
import ufl

import dolfinx as df

from multi.domain import Domain, RectangularDomain
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial
from multi.boundary import plane_at, point_at
from multi.preprocessing import create_meshtags
from multi.product import InnerProduct

from pymor.basic import VectorOperator, StationaryModel
from pymor.bindings.fenicsx import (
    FenicsxVectorSpace,
    FenicsxVisualizer,
    FenicsxMatrixOperator,
)


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


def discretize_subdomain_operators(example):
    from .auxiliary_problem import discretize_auxiliary_problem
    from .matrix_based_operator import FenicsxMatrixBasedOperator

    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, ftags, example.parameters["subdomain"]
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    EMOD = example.youngs_modulus
    POISSON = example.poisson_ratio
    omega = aux.problem.domain
    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d)

    # ### wrap stiffness matrix as pymor operator
    def param_setter(mu):
        d.x.array[:] = 0.0
        aux.solve(d, mu)
        d.x.scatter_forward()

    params = example.parameters["subdomain"]
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs, params, param_setter=param_setter, name="ParaGeom"
    )

    # ### wrap external force as pymor operator
    TY = -example.traction_y
    traction = df.fem.Constant(
        omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(ftags["top"], traction)
    problem.setup_solver()
    problem.assemble_vector(bcs=[])
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))  # type: ignore

    return operator, rhs


def discretize_fom(example, auxiliary_problem, trafo_disp):
    """Discretize FOM with Pull Back"""
    from .fom import ParaGeomLinEla
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCGeom, BCTopo

    domain = auxiliary_problem.problem.domain.grid
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    top_locator = plane_at(example.height, "y")
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)

    EMOD = example.youngs_modulus
    POISSON = example.poisson_ratio
    V = trafo_disp.function_space
    problem = ParaGeomLinEla(omega, V, E=EMOD, NU=POISSON, d=trafo_disp)

    # Dirichlet BCs
    origin = point_at(omega.xmin)
    bottom_right_locator = point_at([omega.xmax[0], omega.xmin[1], 0.0])
    bottom_right = df.mesh.locate_entities_boundary(domain, 0, bottom_right_locator)
    u_origin = df.fem.Constant(domain, (df.default_scalar_type(0.0),) * omega.gdim)
    u_bottom_right = df.fem.Constant(domain, df.default_scalar_type(0.0))

    bc_origin = BCGeom(u_origin, origin, V)
    bc_bottom_right = BCTopo(u_bottom_right, bottom_right, 0, V, sub=1)
    problem.add_dirichlet_bc(
        value=bc_origin.value, boundary=bc_origin.locator, method="geometrical"
    )
    problem.add_dirichlet_bc(
        value=bc_bottom_right.value,
        boundary=bc_bottom_right.entities,
        sub=1,
        method="topological",
        entity_dim=bc_bottom_right.entity_dim,
    )

    # Neumann BCs
    TY = -example.traction_y
    traction = df.fem.Constant(
        domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    problem.setup_solver()
    dirichlet = problem.get_dirichlet_bcs()
    problem.assemble_vector(bcs=dirichlet)

    # ### wrap as pymor model
    def param_setter(mu):
        trafo_disp.x.array[:] = 0.0
        auxiliary_problem.solve(trafo_disp, mu)
        trafo_disp.x.scatter_forward()

    params = example.parameters["global"]
    coeffs = problem.form_lhs.coefficients()  # type: ignore
    assert len(coeffs) == 1  # type: ignore
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs,
        params,
        param_setter=param_setter,
        bcs=(bc_origin, bc_bottom_right),
        name="ParaGeom",
    )

    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem goes out of scope and problem.b is deleted
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))  # type: ignore

    # ### Inner product
    inner_product = InnerProduct(V, product="h1-semi", bcs=dirichlet)
    product_mat = inner_product.assemble_matrix()
    product_name = "h1_0_semi"
    h1_product = FenicsxMatrixOperator(product_mat, V, V, name=product_name)

    # ### Visualizer
    viz = FenicsxVisualizer(FenicsxVectorSpace(V))

    # TODO
    # output functional !!!

    fom = StationaryModel(
        operator, rhs, products={product_name: h1_product}, visualizer=viz, name="FOM"
    )
    return fom
