from mpi4py import MPI
import ufl
import numpy as np
from dolfinx import fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
from dolfinx.io.utils import XDMFFile
from basix.ufl import element
from petsc4py import PETSc

from multi.boundary import plane_at, point_at
from multi.bcs import BoundaryConditions
from multi.domain import Domain
from multi.preprocessing import create_meshtags
from multi.product import InnerProduct
from definitions import Example

from pymor.basic import LincombOperator, VectorOperator, StationaryModel, ProjectionParameterFunctional, VectorFunctional, ExpressionParameterFunctional, GenericParameterFunctional, ConstantOperator
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer


def main():
    ex = Example(name="beam")
    fom = discretize_fom(ex)
    P = fom.parameters.space((1., 2.))
    # solve fom for constant μ and check solution via paraview
    test_mu = P.parameters.parse([1.5 for _ in range(10)])

    data = fom.compute(solution=True, output=True, mu=test_mu)
    U = data.get('solution')
    fom.visualize(U, filename=ex.fom_displacement.as_posix())

    compliance = data.get('output')[0, 0]
    print(f'\nCompliance C(u, mu) = {compliance} for mu={test_mu}')


def discretize_fom(ex):
    """returns FOM as pymor model"""

    # read fine grid from disk
    # with XDMFFile(MPI.COMM_WORLD, ex.fine_grid.as_posix(), "r") as fh:
    with XDMFFile(MPI.COMM_WORLD, ex.fine_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, "subdomains")

    # finite element space
    gdim = domain.ufl_cell().geometric_dimension()
    ve = element("P", domain.basix_cell(), ex.fe_deg, shape=(gdim,))
    V = fem.functionspace(domain, ve)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    num_subdomains = ex.nx * ex.ny

    # create facet tags to define external force
    top_facets = int(99)
    top = {"top": (top_facets, plane_at(1.0, "y"))}
    tdim = domain.topology.dim
    fdim = tdim - 1
    facet_tags, _ = create_meshtags(domain, fdim, top)
    assert facet_tags.find(top_facets).size == ex.nx * ex.resolution

    # Dirichlet BCs
    bc_generator = BoundaryConditions(domain, V, facet_tags=facet_tags)
    omega = Domain(domain)
    origin = point_at(omega.xmin)
    bottom_right = point_at([omega.xmax[0], omega.xmin[1], 0.])
    u_origin = fem.Constant(domain, (default_scalar_type(0.0),) * gdim)
    u_bottom_right = fem.Constant(domain, default_scalar_type(0.0))
    bc_generator.add_dirichlet_bc(u_origin, origin, method="geometrical")
    bc_generator.add_dirichlet_bc(u_bottom_right, bottom_right, sub=1, method="geometrical", entity_dim=0)
    bcs = bc_generator.bcs

    def strain(x):
        """Assume plane strain for ease of implementation"""
        e = ufl.sym(ufl.grad(x))
        return e

    def a_q(subdomain_id, trial, test):
        """ufl form - parameter independent"""
        eps = strain(trial)
        δeps = strain(test)
        i, j = ufl.indices(2)
        E = 20e3 # set reference modulus to 20 GPa, such that μ_i in range (1, 2) to get effective modulus in range (20, 40) GPa
        NU = 0.3
        form = E / (1. + NU) * (
                NU / (1. - 2. * NU) * eps[i, i] * δeps[j, j] # type: ignore
                + eps[i, j] * δeps[i, j] # type: ignore
                ) * dx(subdomain_id)
        return form

    def ass_mat(subdomain_id, bcs):
        """assemble matrix"""
        a = a_q(subdomain_id, u, v)
        cpp_form = fem.form(a, form_compiler_options={}, jit_options={})
        A = assemble_matrix(cpp_form, bcs=bcs, diagonal=0.)
        A.assemble()

        return A

    # assemble matrices
    matrices = []
    for id in range(num_subdomains):
        matrices.append(ass_mat(id, bcs))
    assert matrices[0] is not matrices[1]
    # create matrix to account for BCs
    zero_form = ufl.inner(u, v) * u_bottom_right * dx # type:ignore
    bc_mat = assemble_matrix(fem.form(zero_form), bcs=bcs, diagonal=1.)
    bc_mat.assemble()

    # inner product
    inner_product = InnerProduct(V, product="h1-semi", bcs=bcs)
    product_mat = inner_product.assemble_matrix()

    # external force vector
    # max deflection around omega.xmax[0] / 10 = 1 (rough estimate), needs confirmation
    loading = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(-10.)))
    bc_generator.add_neumann_bc(top_facets, loading)
    f_ext = bc_generator.neumann_bcs
    b = assemble_vector(fem.form(f_ext))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE) # type: ignore
    set_bc(b, bcs)

    # ### wrap everything as pymor model
    parameters = {"E": num_subdomains}
    parameter_functionals = [ProjectionParameterFunctional("E", size=num_subdomains, index=q) for q in range(num_subdomains)]
    ops = [FenicsxMatrixOperator(bc_mat, V, V), ] + [FenicsxMatrixOperator(mat, V, V) for mat in matrices]
    operator = LincombOperator(ops, [1., ] + parameter_functionals)
    F_ext = FenicsxVectorSpace(V).make_array([b]) # type: ignore
    rhs = VectorOperator(F_ext)
    h1_product = FenicsxMatrixOperator(product_mat, V, V, name='h1_0_semi')
    viz = FenicsxVisualizer(FenicsxVectorSpace(V))

    # output: C(u, mu) = f_{ext}^T u(mu) + Σ_i w_i (mu_i - 1) ** 2
    # implemented as LincombOperator
    # note: f_{ext}^T u(mu) = inner(f_ext, u(mu))
    # inner( , ) denotes inner product given by `product`
    compliance = VectorFunctional(F_ext, product=None, name="compliance")

    mid_points = np.linspace(0, 9, num=10) + 0.5
    L = omega.xmax[0]
    weights = mid_points ** 2 - L * mid_points + L ** 2 / 4

    cost = GenericParameterFunctional(lambda mu: np.dot(weights, (mu["E"] - 1.0) ** 2), parameters)
    # always returns 1.
    One = ConstantOperator(compliance.range.ones(1), source=compliance.source)

    objective = LincombOperator([compliance, One], [1., cost])
    # TODO what is the derivative wrt mu?

    fom = StationaryModel(
            operator, rhs, output_functional=objective,
            products={"h1_0_semi": h1_product},
            visualizer=viz
            )

    return fom


if __name__ == "__main__":
    main()
