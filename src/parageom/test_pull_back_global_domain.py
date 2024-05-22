import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio, XDMFFile

from multi.boundary import plane_at, point_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from multi.projection import relative_error, absolute_error

TY = -10.
# material parameters for the test problem
# E and NU for auxiliary problem are defined in function `discretize_auxiliary_problem`
EMOD = 20e3
POISSON = 0.3


def compute_reference_solution(mshfile, degree, d):
    # translate parent domain
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

    x_domain = domain.geometry.x
    disp = np.pad(
        d.x.array.reshape(x_domain.shape[0], -1),  # type: ignore
        pad_width=[(0, 0), (0, 1)],
    )
    x_domain += disp

    # Physical Domain
    top_locator = plane_at(1000.0, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)

    # Problem
    material = LinearElasticMaterial(gdim=omega.gdim, E=EMOD, NU=POISSON)
    V = fem.functionspace(domain, ("P", degree, (omega.gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=material)

    # Dirichlet BCs
    origin = point_at(omega.xmin)
    bottom_right = point_at([omega.xmax[0], omega.xmin[1], 0.])
    u_origin = fem.Constant(domain, (default_scalar_type(0.0),) * omega.gdim)
    u_bottom_right = fem.Constant(domain, default_scalar_type(0.0))
    problem.add_dirichlet_bc(u_origin, origin, method="geometrical")
    problem.add_dirichlet_bc(u_bottom_right, bottom_right, sub=1, method="geometrical", entity_dim=0)

    # Neumann BCs
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    # solve
    problem.setup_solver()
    u = problem.solve()
    return u


def discretize_fom(auxiliary_problem, trafo_disp):
    from .fom import ParaGeomLinEla
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCGeom, BCTopo
    from pymor.basic import VectorOperator, StationaryModel

    domain = auxiliary_problem.problem.domain.grid
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    top_locator = plane_at(1000.0, "y")
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)

    V = trafo_disp.function_space
    problem = ParaGeomLinEla(omega, V, E=EMOD, NU=POISSON, d=trafo_disp)

    # Dirichlet BCs
    origin = point_at(omega.xmin)
    bottom_right_locator = point_at([omega.xmax[0], omega.xmin[1], 0.])
    bottom_right = mesh.locate_entities_boundary(domain, 0, bottom_right_locator)
    u_origin = fem.Constant(domain, (default_scalar_type(0.0),) * omega.gdim)
    u_bottom_right = fem.Constant(domain, default_scalar_type(0.0))

    bc_origin = BCGeom(u_origin, origin, V)
    bc_bottom_right = BCTopo(u_bottom_right, bottom_right, 0, V, sub=1)
    problem.add_dirichlet_bc(value=bc_origin.value, boundary=bc_origin.locator, method="geometrical")
    problem.add_dirichlet_bc(value=bc_bottom_right.value, boundary=bc_bottom_right.entities, sub=1, method="topological", entity_dim=bc_bottom_right.entity_dim)

    # Neumann BCs
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    problem.setup_solver()
    problem.assemble_vector(bcs=problem.get_dirichlet_bcs())

    # ### wrap as pymor model
    def param_setter(mu):
        trafo_disp.x.array[:] = 0.
        auxiliary_problem.solve(trafo_disp, mu)

    params = {"R": 10}
    operator = FenicsxMatrixBasedOperator(problem.form_lhs, params, param_setter=param_setter,
                                          bcs=(bc_origin, bc_bottom_right), name="ParaGeom")

    # NOTE
    # without b.copy(), fom.rhs.as_range_array() does not return correct data
    # problem goes out of scope and problem.b is deleted
    rhs = VectorOperator(operator.range.make_array([problem.b.copy()]))
    fom = StationaryModel(operator, rhs, name="FOM")

    coeffs = problem.form_lhs.coefficients()
    assert len(coeffs) == 1
    assert coeffs[0].name == "d_trafo"

    return fom


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # Generate physical subdomain
    parent_domain_msh = example.global_parent_domain.as_posix()
    degree = example.geom_deg

    interface_tags = [i for i in range(15, 25)] # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        parent_domain_msh, degree, interface_tags, example.parameters["global"]
    )
    values = [150., 170., 190., 210., 230., 250., 270., 290., 200., 300.]
    mu = aux.parameters.parse(values)
    d = fem.Function(aux.problem.V, name="d_trafo")
    aux.solve(d, mu)  # type: ignore
    u_phys = compute_reference_solution(parent_domain_msh, degree, d)

    # u on physical mesh
    with XDMFFile(u_phys.function_space.mesh.comm, "uphys.xdmf", "w") as xdmf:
        xdmf.write_mesh(u_phys.function_space.mesh) # type: ignore
        xdmf.write_function(u_phys, t=0.0) # type: ignore

    fom = discretize_fom(aux, d)
    U = fom.solve(mu)
    u = fem.Function(aux.problem.V)
    u.x.array[:] = U.to_numpy().flatten()

    # compare on reference domain
    breakpoint()
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


if __name__ == "__main__":
    main()
