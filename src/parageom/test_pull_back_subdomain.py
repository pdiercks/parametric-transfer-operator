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


def compute_reference_solution(mshfile, degree, d):
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

    x_subdomain = domain.geometry.x
    disp = np.pad(
        d.x.array.reshape(x_subdomain.shape[0], -1),  # type: ignore
        pad_width=[(0, 0), (0, 1)],
    )
    x_subdomain += disp

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


def discretize_fom(auxiliary_problem, trafo_disp):
    from .fom import ParaGeomLinEla, ParaGeomOperator
    from pymor.basic import VectorOperator, StationaryModel

    top_locator = plane_at(1.0, "y")
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
    problem.add_dirichlet_bc(value=zero, boundary=bottom_locator, method="geometrical")
    problem.add_neumann_bc(top_marker, traction)
    problem.setup_solver()
    # problem.assemble_matrix(bcs=problem.get_dirichlet_bcs())
    problem.assemble_vector(bcs=problem.get_dirichlet_bcs())

    # ### wrap as pymor model
    def param_setter(mu):
        trafo_disp.x.array[:] = 0.
        auxiliary_problem.solve(trafo_disp, mu)

    params = {"R": [1]}
    # not sure how params would map to fem.Constant
    # self.parameters.assert_compatible(mu) needs to eval to True
    operator = ParaGeomOperator(problem.form_lhs, params, param_setter, bcs=problem.get_dirichlet_bcs(), name="ParaGeom_a")
    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem goes out of scope and problem.b is deleted
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))
    fom = StationaryModel(operator, rhs, name="FOM")

    return fom


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # Generate physical subdomain
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, example.parameters["subdomain"]
    )
    mu = aux.parameters.parse([0.29001])
    d = fem.Function(aux.problem.V)
    aux.solve(d, mu)  # type: ignore
    u_phys = compute_reference_solution(parent_subdomain_msh, degree, d)

    fom = discretize_fom(aux, d)
    U = fom.solve(mu)
    u = fem.Function(aux.problem.V)
    u.x.array[:] = U.to_numpy().flatten()

    # compare on reference domain
    V = aux.problem.V
    u_ref = fem.Function(V)
    interpolation_data = fem.create_nonmatching_meshes_interpolation_data(
        V.mesh,
        V.element,
        u_phys.function_space.mesh,
        padding=1e-12,
    )
    u_ref.interpolate(u_phys, nmm_interpolation_data=interpolation_data) # type: ignore

    # u rom on parent domain
    # u physical interpolated onto parent domain
    with XDMFFile(aux.problem.domain.grid.comm, "urom.xdmf", "w") as xdmf:
        xdmf.write_mesh(u.function_space.mesh)
        xdmf.write_function(u, t=0.0)
        # xdmf.write_function(u_ref, t=0.0)

    # u on physical mesh
    with XDMFFile(u_phys.function_space.mesh.comm, "uphys.xdmf", "w") as xdmf:
        xdmf.write_mesh(u_phys.function_space.mesh) # type: ignore
        xdmf.write_function(u_phys, t=0.0) # type: ignore

    inner_product = InnerProduct(V, "mass")
    prod_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(prod_mat, V, V)
    source = FenicsxVectorSpace(V)
    urom = source.make_array([u.vector]) # type: ignore
    uphys = source.make_array([u_phys.vector]) # type: ignore

    abs_err = absolute_error(uphys, urom, product)
    rel_err = relative_error(uphys, urom, product)
    print(f"{abs_err=}")
    print(f"{rel_err=}")

    # ### Option A: EI of the bilinear operator
    # U = fom.solve(mu)
    # evaluations.append(op.apply(U, mu)) # internal force
    # cbasis, ipoints, data = ei_greedy(evaluations)
    # ei_op = ...
    # TODO: restricted evaluation for fom.operator may be more involved
    # project ei_op


if __name__ == "__main__":
    main()
