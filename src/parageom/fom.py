import typing

from mpi4py import MPI
import dolfinx as df
import numpy as np
import basix
import ufl

from multi.domain import Domain, RectangularDomain, StructuredQuadGrid
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial
from multi.preprocessing import create_meshtags
from multi.product import InnerProduct
from multi.io import read_mesh

from pymor.basic import VectorOperator, StationaryModel, GenericParameterFunctional, NumpyVectorSpace, VectorFunctional
from pymor.operators.constructions import ConstantOperator, LincombOperator
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
        d: df.fem.Function,
        matparam: dict,
    ):
        """Initialize linear elastic model with pull back.

        Args:
            domain: The parent domain.
            V: FE space.
            d: parametric transformation displacement field.
            matparam: parameters defining the material.
              See `multi.materials.LinearElasticMaterial`.
        """
        super().__init__(domain, V)
        self.mat = LinearElasticMaterial(**matparam)
        self.d = d
        self.dx = ufl.Measure("dx", domain=domain.grid)

    def displacement_gradient(self, u: typing.Union[ufl.TrialFunction, df.fem.Function]):
        """Returns weighted displacement gradient."""
        F = self.deformation_gradient(self.d)
        Finv = ufl.inv(F)
        grad_u = self.transformation_gradient(u)
        H = ufl.dot(grad_u, Finv)
        if self.mat.plane_stress:
            lame_1 = self.mat.lambda_1
            lame_2 = self.mat.lambda_2
            Hzz = -lame_1 / (2.*lame_2+lame_1) * (H[0, 0] + H[1, 1])
            return ufl.as_tensor([
                [H[0, 0], H[0, 1], 0.0],
                [H[1, 0], H[1, 1], 0.0],
                [0.0, 0.0, Hzz],
                ])
        else:
            return ufl.as_tensor([
                [H[0, 0], H[0, 1], 0.0],
                [H[1, 0], H[1, 1], 0.0],
                [0.0, 0.0, 0.0],
                ])

    def transformation_gradient(self, d: df.fem.Function):
        H = ufl.grad(d)
        return ufl.as_tensor([
            [H[0, 0], H[0, 1], 0.0],
            [H[1, 0], H[1, 1], 0.0],
            [0.0, 0.0, 0.0],
            ])

    def deformation_gradient(self, d: df.fem.Function):
        Id = ufl.Identity(3)
        F = Id + self.transformation_gradient(d) # type: ignore
        return F

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
        Id = ufl.Identity(3)

        i, j, k, l = ufl.indices(4) # noqa
        tetrad_ijkl = lame_1 * Id[i, j] * Id[k, l] + lame_2 * (
            Id[i, k] * Id[j, l] + Id[i, l] * Id[j, k]  # type: ignore
        )
        grad_u_kl = self.displacement_gradient(w)[k, l] # type: ignore
        sigma_ij = ufl.as_tensor(tetrad_ijkl * grad_u_kl, (i, j))  # type: ignore
        return sigma_ij

    @property
    def form_volume(self):
        F = self.deformation_gradient(self.d)
        detF = ufl.det(F)
        c = df.fem.Constant(self.domain.grid, df.default_scalar_type(1.))
        return c * detF * self.dx # type: ignore

    @property
    def form_lhs(self):
        grad_v_ij = self.displacement_gradient(self.test)
        sigma_ij = self.weighted_stress(self.trial)
        F = self.deformation_gradient(self.d)
        detF = ufl.det(F)
        return ufl.inner(grad_v_ij, sigma_ij) * detF * self.dx # type: ignore

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
    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        example, parent_subdomain_msh, ftags, example.parameters["subdomain"]
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    omega = aux.problem.domain
    matparam = {"gdim": omega.gdim, "E": 1.0, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    problem = ParaGeomLinEla(omega, aux.problem.V, d, matparam)

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
    EMOD = example.youngs_modulus
    TY = -example.traction_y / EMOD
    traction = df.fem.Constant(
        omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(ftags["top"], traction)
    problem.setup_solver()
    problem.assemble_vector(bcs=[])
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))  # type: ignore

    return operator, rhs


def discretize_fom(example: BeamData, auxiliary_problem, trafo_disp, ω=0.5):
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
    V = trafo_disp.function_space

    # ### Initialize global coarse grid
    # required for definition of Dirichlet & Neumann BCs
    grid, _, _ = read_mesh(
        example.coarse_grid("global"),
        MPI.COMM_SELF,
        kwargs={"gdim": example.gdim},
    )
    coarse_grid = StructuredQuadGrid(grid)

    # Neumann BCs
    neumann = example.get_neumann(coarse_grid.grid, "left")
    assert neumann is not None
    facet_tags, _ = create_meshtags(domain, fdim, {"top": neumann})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    matparam = {"gdim": omega.gdim, "E": 1.0, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    problem = ParaGeomLinEla(omega, auxiliary_problem.problem.V, trafo_disp, matparam)

    # Dirichlet BCs
    dirichlet_left = example.get_dirichlet(coarse_grid.grid, "left")
    dirichlet_right = example.get_dirichlet(coarse_grid.grid, "right")
    assert dirichlet_left is not None
    assert dirichlet_right is not None

    dirichlet_bcs = []
    for bc_spec in dirichlet_left:
        entities_left = df.mesh.locate_entities_boundary(
            domain, bc_spec["entity_dim"], bc_spec["boundary"]
        )
        assert entities_left.size > 0
        bc_left = BCTopo(
            df.fem.Constant(V.mesh, bc_spec["value"]),
            entities_left,
            bc_spec["entity_dim"],
            V,
            sub=bc_spec["sub"],
        )
        dirichlet_bcs.append(bc_left)

    for bc_spec in dirichlet_right:
        entities_right = df.mesh.locate_entities_boundary(
            domain, bc_spec["entity_dim"], bc_spec["boundary"]
        )
        assert entities_right.size > 0
        bc_right = BCTopo(
            df.fem.Constant(V.mesh, bc_spec["value"]),
            entities_right,
            bc_spec["entity_dim"],
            V,
            sub=bc_spec["sub"],
        )
        dirichlet_bcs.append(bc_right)

    bcs = _create_dirichlet_bcs(tuple(dirichlet_bcs))

    # Neumann BCs
    EMOD = example.youngs_modulus
    TY = -example.traction_y / EMOD
    traction = df.fem.Constant(
        domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(neumann[0], traction)

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
        bcs=tuple(dirichlet_bcs),
        name="ParaGeom",
    )

    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem goes out of scope and problem.b is deleted
    F_ext = operator.range.make_array([problem.b.copy()]) # type: ignore
    rhs = VectorOperator(F_ext)  # type: ignore

    # ### Inner products
    inner_product = InnerProduct(V, product="h1-semi", bcs=bcs)
    product_mat = inner_product.assemble_matrix()
    product_name = "h1_0_semi"
    h1_product = FenicsxMatrixOperator(product_mat, V, V, name=product_name)

    l2_inner = InnerProduct(V, product="l2", bcs=bcs)
    product_mat = l2_inner.assemble_matrix()
    product_l2 = "l2"
    l2_product = FenicsxMatrixOperator(product_mat, V, V, name=product_l2)

    # ### Visualizer
    viz = FenicsxVisualizer(FenicsxVectorSpace(V))

    def compute_volume(mu):
        param_setter(mu)

        vcpp = df.fem.form(problem.form_volume)
        vol = df.fem.assemble_scalar(vcpp) # type: ignore
        return vol

    # ### Output definition
    # solve FOM for initial mu
    initial_mu = params.parse([0.1 for _ in range(example.nx)])
    vol_ref = compute_volume(initial_mu)
    U_ref = operator.apply_inverse(rhs.as_range_array(), mu=initial_mu)

    # mass/volume
    volume = GenericParameterFunctional(compute_volume, params)
    vol_va = NumpyVectorSpace(1).ones(1)
    vol_va.scal( (1. - ω) / vol_ref)
    one_op = ConstantOperator(vol_va, source=operator.source)

    # compliance
    compl_ref = F_ext.inner(U_ref).item()
    scaled_fext = F_ext.copy()
    scaled_fext.scal(1 / compl_ref)
    compliance = VectorFunctional(scaled_fext, product=None, name="compliance")

    # output J = (1 - ω) mass + ω compliance
    output = LincombOperator([one_op, compliance], [volume, ω])

    fom = StationaryModel(
        operator,
        rhs,
        output_functional=output,
        products={product_name: h1_product, product_l2: l2_product},
        visualizer=viz,
        name="FOM",
    )
    return fom


if __name__ == "__main__":
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .stress_analysis import principal_stress_2d, project

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
        [0.3 * example.unit_length for _ in range(10)]
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
    assert np.isclose(total_load, -example.traction_y / example.youngs_modulus)

    u = df.fem.Function(auxp.problem.V)
    mesh = u.function_space.mesh
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    V = fom.solution_space.V
    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    q_degree = 2
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(V.mesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    matparam = {"gdim": 2, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom_physical = ParaGeomLinEla(auxp.problem.domain, auxp.problem.V, d, matparam)

    # scalar quadrature space
    qs = basix.ufl.quadrature_element(basix_celltype, value_shape=(2,), scheme="default", degree=q_degree)
    Q = df.fem.functionspace(V.mesh, qs)
    sigma_q = df.fem.Function(Q, name="sp")
    W = df.fem.functionspace(V.mesh, ("P", 2, (2,))) # target space, linear Lagrange elements, store both s1 and s2 as components
    sigma_p = df.fem.Function(W, name="sp")


    tdim = V.mesh.topology.dim
    map_c = V.mesh.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    f_quad = lambda x: (0.1 - 0.3) / 81 * x ** 2 + 0.3 # noqa

    designs = {
            "max_vol": fom.parameters.parse([0.1 for _ in range(10)]),
            "reference": fom.parameters.parse([0.2 for _ in range(10)]),
            "min_vol": fom.parameters.parse([0.3 for _ in range(10)]),
            "random": parameter_space.sample_randomly(1)[0],
            "linear": fom.parameters.parse(np.linspace(0.3, 0.1, num=10)),
            "quadratic": fom.parameters.parse(f_quad(np.linspace(0, 10, num=10, endpoint=False))),
            "optimum": fom.parameters.parse([0.28793595814454276, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
            }
    for name, mu in designs.items():
        u.x.array[:] = 0.0
        stress.x.array[:] = 0.0
        sigma_q.x.array[:] = 0.0

        U = fom.solve(mu)
        # fom.visualize(U, filename=f"output/{name}_u.xdmf")
        u.x.array[:] = U.to_numpy().flatten()
        s1, s2 = principal_stress_2d(u, parageom_physical, q_degree, cells, stress.x.array.reshape(cells.size, -1))

        print(f"{name=}")
        print(f"{s1.flatten().min()} <= s1 <= {s1.flatten().max()}")
        print(f"{s2.flatten().min()} <= s2 <= {s2.flatten().max()}")
        sigma_q.x.array[::2] = s1.flatten()
        sigma_q.x.array[1::2] = s2.flatten()
        project(sigma_q, sigma_p)

        # with df.io.XDMFFile(W.mesh.comm, f"output/{name}_s.xdmf", "w") as xdmf:
        #     xdmf.write_mesh(W.mesh)
        #     xdmf.write_function(sigma_p)
