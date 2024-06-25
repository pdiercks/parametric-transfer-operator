import typing
import ufl

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np

from multi.domain import Domain, RectangularDomain, StructuredQuadGrid
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial
from multi.boundary import plane_at
from multi.preprocessing import create_meshtags
from multi.product import InnerProduct
from multi.io import read_mesh

from pymor.basic import VectorOperator, StationaryModel
from pymor.bindings.fenicsx import (
    FenicsxVectorSpace,
    FenicsxVisualizer,
    FenicsxMatrixOperator,
)
from .definitions import BeamData

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
    # problem = ParaGeomLinEla(omega, aux.problem.V, E=1.0, NU=POISSON, d=d)
    problem = ParaGeomLinEla(omega, aux.problem.V, E=EMOD, NU=POISSON, d=d)

    # ### wrap stiffness matrix as pymor operator
    def param_setter(mu):
        print(f"---- SOLVING AUXILIARY PROBLEM ---- {mu=}")
        d.x.array[:] = 0.0
        aux.solve(d, mu)
        d.x.scatter_forward()

    params = example.parameters["subdomain"]
    operator = FenicsxMatrixBasedOperator(
        problem.form_lhs, params, param_setter=param_setter, name="ParaGeom"
    )

    # ### wrap external force as pymor operator
    # TY = -example.traction_y / EMOD
    TY = -example.traction_y * 100.
    traction = df.fem.Constant(
        omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(ftags["top"], traction)
    problem.setup_solver()
    problem.assemble_vector(bcs=[])
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))  # type: ignore

    return operator, rhs


def discretize_fom(example: BeamData, auxiliary_problem, trafo_disp):
    """Discretize FOM with Pull Back"""
    from .fom import ParaGeomLinEla
    from .matrix_based_operator import (
        FenicsxMatrixBasedOperator,
        BCTopo,
        _create_dirichlet_bcs,
    )

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
    problem = ParaGeomLinEla(omega, V, E=1.0, NU=POISSON, d=trafo_disp)

    # ### Dirichlet bcs are defined globally
    grid, _, _ = read_mesh(
        example.coarse_grid("global"),
        MPI.COMM_SELF,
        kwargs={"gdim": example.gdim},
    )
    coarse_grid = StructuredQuadGrid(grid)
    dirichlet_left = example.get_dirichlet(coarse_grid.grid, "left")
    dirichlet_right = example.get_dirichlet(coarse_grid.grid, "right")
    assert dirichlet_left is not None
    assert dirichlet_right is not None

    # Dirichlet BCs
    entities_left = df.mesh.locate_entities_boundary(
        domain, fdim, dirichlet_left["boundary"]
    )
    bc_left = BCTopo(
        df.fem.Constant(V.mesh, dirichlet_left["value"]),
        entities_left,
        fdim,
        V,
        sub=dirichlet_left["sub"],
    )
    entities_right = df.mesh.locate_entities_boundary(
        domain, fdim, dirichlet_right["boundary"]
    )
    bc_right = BCTopo(
        df.fem.Constant(V.mesh, dirichlet_right["value"]),
        entities_right,
        fdim,
        V,
        sub=dirichlet_right["sub"],
    )
    bcs = _create_dirichlet_bcs((bc_left, bc_right))

    # Neumann BCs
    TY = -example.traction_y / EMOD
    traction = df.fem.Constant(
        domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    problem.setup_solver()
    problem.assemble_vector(bcs=bcs)

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
        bcs=(bc_left, bc_right),
        name="ParaGeom",
    )

    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem goes out of scope and problem.b is deleted
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))  # type: ignore

    # ### Inner products
    inner_product = InnerProduct(V, product="h1-semi", bcs=bcs)
    product_mat = inner_product.assemble_matrix()
    product_name = "h1_0_semi"
    h1_product = FenicsxMatrixOperator(product_mat, V, V, name=product_name)

    l2_inner = InnerProduct(V, product="l2", bcs=bcs)
    product_mat = l2_inner.assemble_matrix()
    product_l2 = "l2"
    l2_product = FenicsxMatrixOperator(product_mat, V, V, name=product_l2)

    # u = ufl.TrialFunction(V)
    # v = ufl.TestFunction(V)
    # l_char = df.fem.Constant(V.mesh, df.default_scalar_type(100. ** 2))
    # scaled_h1_0_semi = l_char * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    # a_cpp = df.fem.form(scaled_h1_0_semi)
    # A = dolfinx.fem.petsc.create_matrix(a_cpp)
    # A.zeroEntries()
    # dolfinx.fem.petsc.assemble_matrix(A, a_cpp, bcs=bcs)
    # A.assemble()
    # scaled_h1_product = FenicsxMatrixOperator(A, V, V, name="scaled_h1_0_semi")

    # scaled_l2 = l_char * ufl.inner(u, v) * ufl.dx
    # l2_cpp = df.fem.form(scaled_l2)
    # P = dolfinx.fem.petsc.create_matrix(l2_cpp)
    # P.zeroEntries()
    # dolfinx.fem.petsc.assemble_matrix(P, l2_cpp, bcs=bcs)
    # P.assemble()
    # scaled_l2_product = FenicsxMatrixOperator(P, V, V, name="scaled_l2")

    # ### Visualizer
    viz = FenicsxVisualizer(FenicsxVectorSpace(V))

    # TODO
    # output functional !!!

    fom = StationaryModel(
        operator,
        rhs,
        products={product_name: h1_product, product_l2: l2_product},
        visualizer=viz,
        name="FOM",
    )
    return fom


if __name__ == "__main__":
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    degree = example.geom_deg
    interface_tags = [
        i for i in range(15, 25)
    ]  # FIXME better define in Example data class
    auxp = discretize_auxiliary_problem(
        parent_domain_path,
        degree,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    d = df.fem.Function(auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, auxp, d)

    parameter_space = auxp.parameters.space(example.mu_range)
    mu = parameter_space.parameters.parse(
        [0.2 * example.unit_length for _ in range(10)]
    )
    U = fom.solve(mu) # dimensionless solution U, real displacement D=l_char * U
    # with characteristic length l_char = 100. mm (unit length)

    D = U.copy()
    l_char = 100.
    D.scal(l_char)

    # check norm of displacement field
    assert np.isclose(D.norm(), U.norm() * l_char)
    assert np.isclose(D.norm(fom.h1_0_semi_product), U.norm(fom.h1_0_semi_product) * l_char)

    # check load
    total_load = np.sum(fom.rhs.as_range_array().to_numpy())  # type: ignore
    assert np.isclose(total_load, -example.traction_y / example.youngs_modulus * 10)

    fom.visualize(U, filename="fom_mu_bar.xdmf")
