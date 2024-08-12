import numpy as np

from mpi4py import MPI
import dolfinx as df
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
import ufl
import basix

from multi.boundary import plane_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct
from multi.io import read_mesh

from pymor.bindings.fenicsx import FenicsxMatrixOperator
from multi.projection import relative_error, absolute_error
from .definitions import BeamData


def compute_reference_solution(example: BeamData, mshfile, degree, d):
    from .matrix_based_operator import BCTopo, _create_dirichlet_bcs
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
    V = df.fem.functionspace(domain, ("P", degree, (omega.gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=material)

    # Dirichlet BCs
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
    dirichlet_bcs = []
    for spec in dirichlet_left + dirichlet_right:
        entities = df.mesh.locate_entities_boundary(domain, fdim, spec["boundary"])
        bc = BCTopo(df.fem.Constant(V.mesh, spec["value"]), entities, fdim, V, sub=spec["sub"])
        dirichlet_bcs.append(bc)

    bcs = _create_dirichlet_bcs(tuple(dirichlet_bcs))
    for bc in bcs:
        problem.add_dirichlet_bc(bc)

    # Neumann BCs
    TY = -example.traction_y
    traction = df.fem.Constant(
        domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    # solve
    problem.setup_solver()
    u = problem.solve()

    # stress analysis
    mesh = u.function_space.mesh
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    q_degree = 2
    basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(3,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(V.mesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    σ = material.sigma(u) # type: ignore
    sigma_voigt = ufl.as_vector([σ[0, 0], σ[1, 1], σ[0, 1]])
    stress_expr = df.fem.Expression(sigma_voigt, q_points)
    stress_expr.eval(domain, entities=cells, values=stress.x.array.reshape(cells.size, -1))

    return u, stress


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, ParaGeomLinEla

    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    degree = example.geom_deg

    interface_tags = [i for i in range(15, 25)] # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        parent_domain_path, degree, interface_tags, example.parameters["global"], coarse_grid=coarse_grid_path
    )
    parameter_space = aux.parameters.space(example.mu_range)
    # mu = parameter_space.sample_randomly(1)[0]
    mu = parameter_space.parameters.parse([0.2 for _ in range(10)])
    d = df.fem.Function(aux.problem.V, name="d_trafo")
    fom = discretize_fom(example, aux, d)
    U = fom.solve(mu)

    # ### reference displacement solution on the physical mesh
    u_phys, stress_phys = compute_reference_solution(example, parent_domain_path, degree, d)

    # u on physical mesh
    with XDMFFile(u_phys.function_space.mesh.comm, "uphys.xdmf", "w") as xdmf:
        xdmf.write_mesh(u_phys.function_space.mesh) # type: ignore
        xdmf.write_function(u_phys, t=0.0) # type: ignore

    # fom.visualize(U, filename="test_fom.bp")

    # ISSUE
    # interpolation does not work to compare the two, because
    # it works based on geometry.
    # In the physical mesh, the whole might actually be bigger/smaller and therefore
    # for some cells in the parent mesh there are no cells at these positions.
    # Therefore, after the interpolation the values at these locations are simply zero.
    source = fom.solution_space
    U_ref = source.from_numpy(u_phys.x.array.reshape(1, -1)) # type: ignore
    fom.visualize(U_ref, filename="test_fom_pull_back.xdmf")

    inner_product = InnerProduct(source.V, "h1")
    prod_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(prod_mat, source.V, source.V)

    abs_err = absolute_error(U_ref, U, product)
    rel_err = relative_error(U_ref, U, product)

    mesh = u_phys.function_space.mesh
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    def compute_principal_components(f):
        values = f.reshape(cells.size, 4, 3)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 2]
        fmin = (fxx+fyy) / 2 - np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        fmax = (fxx+fyy) / 2 + np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        return fmin, fmax

    def risk_factor(s):
        lower_bound = -20.0
        upper_bound = 2.2
        compression = np.max(s / lower_bound)
        tension = np.max(s / upper_bound)
        return 1. - max(compression, tension)

    s1, s2 = compute_principal_components(stress_phys.x.array)
    # print(risk_factor(s1))
    # print(risk_factor(s2))

    parageom = ParaGeomLinEla(aux.problem.domain, d.function_space, E=example.youngs_modulus, NU=example.poisson_ratio, d=d)
    u_parent = df.fem.Function(d.function_space)
    u_parent.x.array[:] = U.to_numpy().flatten()
    ws = parageom.weighted_stress(u_parent)
    sigma_voigt = ufl.as_vector([ws[0, 0], ws[1, 1], ws[0, 1]])

    q_degree = 2
    basix_celltype = getattr(basix.CellType, d.function_space.mesh.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(3,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(d.function_space.mesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    stress_expr = df.fem.Expression(sigma_voigt, q_points)
    stress_expr.eval(d.function_space.mesh, entities=cells, values=stress.x.array.reshape(cells.size, -1))

    ws1, ws2 = compute_principal_components(stress.x.array)
    # print(risk_factor(ws1))
    # print(risk_factor(ws2))

    stress_error = stress_phys.x.array - stress.x.array
    stress_error_relative = stress_error / stress_phys.x.array

    print(f"""\nSummary
          Displacement Error
          ==================
          absolute: {abs_err}
          relative: {rel_err}

          Stress Error
          ============
          absolute: {np.linalg.norm(stress_error)}
          relative: {np.linalg.norm(stress_error_relative)}""")


if __name__ == "__main__":
    main()
