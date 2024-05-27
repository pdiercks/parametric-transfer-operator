"""Build single domain ROM"""

from collections import defaultdict
from mpi4py import MPI  
import dolfinx as df
from dolfinx.io import gmshio

from pymor.algorithms.pod import pod
from pymor.reductors.basic import StationaryRBReductor
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer

from multi.boundary import plane_at, point_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct

import numpy as np


def discretize_problem(example, mshfile):
    # translate parent domain
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

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
    assert example.fe_deg == example.geom_deg
    degree = example.fe_deg
    V = df.fem.functionspace(domain, ("P", degree, (omega.gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=material)

    # Dirichlet BCs
    origin = point_at(omega.xmin)
    bottom_right = point_at([omega.xmax[0], omega.xmin[1], 0.])
    u_origin = df.fem.Constant(domain, (df.default_scalar_type(0.0),) * omega.gdim)
    u_bottom_right = df.fem.Constant(domain, df.default_scalar_type(0.0))
    problem.add_dirichlet_bc(u_origin, origin, method="geometrical")
    problem.add_dirichlet_bc(u_bottom_right, bottom_right, sub=1, method="geometrical", entity_dim=0)

    # Neumann BCs
    TY = -example.traction_y
    traction = df.fem.Constant(
        domain, (df.default_scalar_type(0.0), df.default_scalar_type(TY))
    )
    problem.add_neumann_bc(top_marker, traction)

    problem.setup_solver()
    return problem


def physical_mesh_rom():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .lhs import sample_lhs

    # ### Auxiliary problem
    coarse_grid_path = example.coarse_grid("global").as_posix()
    global_parent_domain_msh = example.global_parent_domain.as_posix()
    degree = example.geom_deg
    interface_tags = [i for i in range(15, 25)] # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        global_parent_domain_msh, degree, interface_tags, example.parameters["global"], coarse_grid=coarse_grid_path
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")

    # ### FOM (without pull back)
    problem = discretize_problem(example, global_parent_domain_msh)
    solver = problem.solver
    u = df.fem.Function(problem.V)
    bcs = problem.get_dirichlet_bcs()

    # ### Inner product
    inner_product = InnerProduct(problem.V, product="h1-semi", bcs=bcs)
    product_mat = inner_product.assemble_matrix()
    product_name = "h1_0_semi"
    h1_product = FenicsxMatrixOperator(product_mat, problem.V, problem.V, name=product_name)

    parameter_space = aux.parameters.space(example.mu_range)
    random_seed = example.training_set_seed
    num_snapshots = 100
    training_set = sample_lhs(parameter_space, name="R", samples=num_snapshots, criterion="center", random_state=random_seed)

    snapshots = []
    for mu in training_set:

        print(f"Solving FOM for {mu=}")
        # translate domain
        aux.solve(d, mu)
        disp = np.pad(
            d.x.array.reshape(problem.domain._x.shape[0], -1),  # type: ignore
            pad_width=[(0, 0), (0, 1)],
        )
        problem.domain.translate(disp)

        # solve problem
        problem.assemble_vector(bcs=bcs)
        problem.assemble_matrix(bcs=bcs)
        solver.solve(problem.b, u.vector)
        u.x.scatter_forward()

        snapshots.append(u.vector.copy())

        # translate back to origin
        problem.domain.translate(-disp)

    source = FenicsxVectorSpace(problem.V)
    viz = FenicsxVisualizer(source)
    snaps = source.make_array(snapshots)
    basis, svals = pod(snaps, product=h1_product, l2_err=1e-3)

    viz.visualize(snaps, filename="physical_snapshots.xdmf")
    viz.visualize(basis, filename="physical_basis.xdmf")
    breakpoint()


def pull_back_rom():
    from .tasks import example
    from .fom import discretize_fom
    from .auxiliary_problem import discretize_auxiliary_problem
    from .lhs import sample_lhs

    # ### FOM
    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.global_parent_domain.as_posix()
    degree = example.geom_deg
    interface_tags = [
        i for i in range(15, 25)
    ]  # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        parent_domain_path,
        degree,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")
    fom = discretize_fom(example, aux, d)

    # ### POD basis construction
    parameter_space = aux.parameters.space(example.mu_range)
    random_seed = example.training_set_seed
    num_snapshots = 100
    training_set = sample_lhs(parameter_space, name="R", samples=num_snapshots, criterion="center", random_state=random_seed)
    snapshots = fom.solution_space.empty(reserve=num_snapshots)
    for mu in training_set:
        snapshots.append(fom.solve(mu))
    basis, svals = pod(snapshots, product=fom.h1_0_semi_product, rtol=1e-5)
    reductor = StationaryRBReductor(
        fom, product=fom.h1_0_semi_product, check_orthonormality=False
    )
    reductor.extend_basis(basis, method="trivial")

    fom.visualizer.visualize(
        basis, filename="./work/parageom/monolithic_pod_basis.xdmf"
    )
    rom = reductor.reduce()

    errors = defaultdict(list)
    ntest = 10
    test_set = parameter_space.sample_randomly(ntest)
    for mu in test_set:
        fom_data = fom.compute(solution=True, output=False, mu=mu)
        rom_data = rom.compute(solution=True, output=False, mu=mu)
        for key in ("solution",):
            if key == "solution":
                ERR = fom_data.get(key) - reductor.reconstruct(rom_data.get(key))
                err = ERR.norm(fom.h1_0_semi_product)
            else:
                ERR = fom_data.get(key) - rom_data.get(key)
                err = ERR[0, 0]

            errors[key].append(err)

    for key in ("solution",):
        print(
            f"Max {key} error = {max(errors[key])} over test set of size {len(test_set)}"
        )


if __name__ == "__main__":
    # pull_back_rom()
    physical_mesh_rom()
