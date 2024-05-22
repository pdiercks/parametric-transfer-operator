import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio, XDMFFile

from multi.boundary import plane_at, point_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct

from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from multi.projection import relative_error, absolute_error


def compute_reference_solution(example, mshfile, degree, d):
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
    top_locator = plane_at(example.height, "y")
    tdim = domain.topology.dim
    fdim = tdim - 1
    top_marker = int(194)
    facet_tags, _ = create_meshtags(domain, fdim, {"top": (top_marker, top_locator)})
    omega = RectangularDomain(domain, facet_tags=facet_tags)

    # Problem
    EMOD = example.youngs_modulus
    POISSON = example.poisson_ratio
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
    TY = -example.traction_y
    traction = fem.Constant(
        domain, (default_scalar_type(0.0), default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    # solve
    problem.setup_solver()
    u = problem.solve()
    return u


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom

    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.global_parent_domain.as_posix()
    degree = example.geom_deg

    interface_tags = [i for i in range(15, 25)] # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        parent_domain_path, degree, interface_tags, example.parameters["global"], coarse_grid=coarse_grid_path
    )
    values = [150., 170., 190., 210., 230., 250., 270., 290., 200., 300.]
    mu = aux.parameters.parse(values)
    d = fem.Function(aux.problem.V, name="d_trafo")
    aux.solve(d, mu)  # type: ignore
    u_phys = compute_reference_solution(example, parent_domain_path, degree, d)

    # u on physical mesh
    with XDMFFile(u_phys.function_space.mesh.comm, "uphys.xdmf", "w") as xdmf:
        xdmf.write_mesh(u_phys.function_space.mesh) # type: ignore
        xdmf.write_function(u_phys, t=0.0) # type: ignore

    fom = discretize_fom(example, aux, d)
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

    inner_product = InnerProduct(V, "h1")
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
