import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio, XDMFFile

from multi.boundary import plane_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from multi.projection import relative_error, absolute_error

TY = -1000.
# material parameters for the test problem
# E and NU for auxiliary problem are defined in function `discretize_auxiliary_problem`
EMOD = 20e3
POISSON = 0.3


def compute_reference_solution(example, mshfile, degree, d):
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

    x_subdomain = domain.geometry.x
    disp = np.pad(
        d.x.array.reshape(x_subdomain.shape[0], -1),  # type: ignore
        pad_width=[(0, 0), (0, 1)],
    )
    x_subdomain += disp

    top_locator = plane_at(example.height, "y")
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

    # K = csr_array(problem.A.getValuesCSR()[::-1])
    # kappa = np.linalg.norm(K.todense())
    return u


def discretize_fom(example, auxiliary_problem, trafo_disp):
    from .fom import ParaGeomLinEla
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCGeom
    from pymor.basic import VectorOperator, StationaryModel

    top_locator = plane_at(example.height, "y")
    bottom_locator = plane_at(0.0, "y")
    domain = auxiliary_problem.problem.domain.grid
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)
    zero = fem.Constant(domain, (default_scalar_type(0.0), default_scalar_type(0.0)))
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )
    V = trafo_disp.function_space

    problem = ParaGeomLinEla(omega, V, E=EMOD, NU=POISSON, d=trafo_disp)
    bc_bottom = BCGeom(zero, bottom_locator, V)
    problem.add_dirichlet_bc(value=bc_bottom.value, boundary=bc_bottom.locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    problem.assemble_vector(bcs=problem.get_dirichlet_bcs())

    # ### wrap as pymor model
    def param_setter(mu):
        trafo_disp.x.array[:] = 0.
        auxiliary_problem.solve(trafo_disp, mu)

    params = {"R": 1}
    operator = FenicsxMatrixBasedOperator(problem.form_lhs, params, param_setter=param_setter,
                                          bcs=(bc_bottom,), name="ParaGeom")

    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem object goes out of scope and problem.b is deleted
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))
    fom = StationaryModel(operator, rhs, name="FOM")

    assert len(problem.form_lhs.coefficients()) == 1
    coeff = problem.form_lhs.coefficients()[0]
    assert coeff.name == "d_trafo"

    return fom


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # Generate physical subdomain
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, ftags, example.parameters["subdomain"]
    )
    d = fem.Function(aux.problem.V, name="d_trafo")
    mu = aux.parameters.parse([example.unit_length * 0.2511])
    aux.solve(d, mu)  # type: ignore
    u_phys = compute_reference_solution(example, parent_subdomain_msh, degree, d)

    fom = discretize_fom(example, aux, d)
    U = fom.solve(mu)

    V = aux.problem.V
    inner_product = InnerProduct(V, "mass")
    prod_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(prod_mat, V, V)
    source = FenicsxVectorSpace(V)
    Uref = source.from_numpy(u_phys.x.array.reshape(1, -1)) # type: ignore

    abs_err = absolute_error(Uref, U, product)
    rel_err = relative_error(Uref, U, product)
    print(f"{abs_err=}")
    print(f"{rel_err=}")


if __name__ == "__main__":
    main()
