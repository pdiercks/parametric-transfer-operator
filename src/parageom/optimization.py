"""Solution of an optimization problem"""

# NOTE
# the below code was written based on the pyMOR Tutorial
# "Model order reduction for PDE-constrained optimization problems"
# See https://docs.pymor.org/2023-2-0/tutorial_optimization.html

import dolfinx as df
import basix

import numpy as np
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import VectorOperator, LincombOperator, VectorFunctional, ConstantOperator
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.models.basic import StationaryModel
from pymor.core.pickle import dump


def main(args):
    """solve optimization problem for different models"""
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, ParaGeomLinEla

    # TODO add logger

    # ### Build FOM
    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    interface_tags = [
        i for i in range(15, 25)
    ]
    auxiliary_problem = discretize_auxiliary_problem(
        example,
        parent_domain_path,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    V = auxiliary_problem.problem.V
    d_trafo = df.fem.Function(V, name="d_trafo")
    fom = discretize_fom(example, auxiliary_problem, d_trafo, ω=args.omega)

    # ### Build localized ROM
    rom, rec_data = build_localized_rom(args, example, auxiliary_problem, d_trafo, fom.parameters, ω=args.omega)

    # ### Global Function for displacement solution
    displacement = df.fem.Function(V, name="u")

    # ### Dict's to gather minimization data
    fom_minimization_data = {
        "num_evals": 0,
        "evaluations": [],
        "evaluation_points": [],
        "time": np.inf,
    }

    rom_minimization_data = {
        "num_evals": 0,
        "evaluations": [],
        "evaluation_points": [],
        "time": np.inf,
    }

    # ParaGeomLinEla for stress computation
    gdim = fom.solution_space.V.mesh.geometry.dim
    matparam = {"gdim": gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(auxiliary_problem.problem.domain, V, d_trafo, matparam)

    # ### Solve optimization problem using FOM
    num_subdomains = example.nx * example.ny
    initial_guess = fom.parameters.parse([0.1 for _ in range(num_subdomains)])
    parameter_space = fom.parameters.space(example.mu_range)
    mu_range = parameter_space.ranges["R"]
    opt_fom_result = solve_optimization_problem(
        args, initial_guess, mu_range, displacement, fom, parageom, fom_minimization_data, gradient=False, rec_data=None
    )
    mu_ref = opt_fom_result.x
    fom_minimization_data["num_iter"] = opt_fom_result.nit
    fom_minimization_data["mu_min"] = mu_ref
    fom_minimization_data["J(mu_min)"] = opt_fom_result.fun
    fom_minimization_data["status"] = opt_fom_result.status

    displacement.x.array[:] *= 0. # type: ignore
    d_trafo.x.array[:] *= 0. # type: ignore

    # ### Solve Optimization Problem using ROM
    opt_rom_result = solve_optimization_problem(
            args, initial_guess, mu_range, displacement, rom, parageom, rom_minimization_data,
            gradient=False, rec_data=rec_data)
    rom_minimization_data["num_iter"] = opt_rom_result.nit
    rom_minimization_data["mu_N_min"] = opt_rom_result.x
    rom_minimization_data["J_N(mu_N_min)"] = opt_rom_result.fun
    rom_minimization_data["status"] = opt_rom_result.status
    rom_minimization_data["abs_err_mu"] = np.linalg.norm(opt_rom_result.x - mu_ref)
    rom_minimization_data["abs_err_J"] = abs(opt_rom_result.fun - opt_fom_result.fun)

    # FOM output evaluated at mu_min found by ROM
    mu_N_min = fom.parameters.parse(opt_rom_result.x)
    J_mu_N_min = fom.output(mu_N_min)[0, 0] # type: ignore
    rom_minimization_data["J(mu_N_min)"] = J_mu_N_min
    rom_minimization_data["suboptimality"] = abs(J_mu_N_min - opt_fom_result.fun) / opt_fom_result.fun

    print("\nResult of optimization with FOM and FD")
    report(opt_fom_result, fom.parameters.parse, fom_minimization_data)

    print("\nResult of optimization with ROM and FD")
    report(
        opt_rom_result, fom.parameters.parse, rom_minimization_data, reference_mu=mu_ref
    )

    # for k, data in enumerate([fom_minimization_data, rom_minimization_data]):
    #     jvalues = data["evaluations"]
    #     mus = data["evaluation_points"]
    #     price = get_prices(mus)
    #     compl = np.array(jvalues) - np.array(price)
    #     iters = np.arange(data["num_evals"])
    #
    #     data["prices"] = price
    #     data["compliance"] = compl
    #     data["iterations"] = iters

    # if args.show:
    #     import matplotlib.pyplot as plt
    #
    #     fig = plt.figure(constrained_layout=True)
    #     axes = fig.subplots(nrows=2, ncols=1, sharex=True)
    #
    #     fig.suptitle("Minimization of the objective functional J")
    #     for k, data in enumerate([fom_minimization_data, rom_minimization_data]):
    #         iters = data["iterations"]
    #         price = data["prices"]
    #         compl = data["compliance"]
    #         jvalues = data["evaluations"]
    #
    #         axes[k].plot(iters, price, "b-", label="price P")  # type: ignore
    #         axes[k].plot(iters, compl, "r-", label="compliance C")  # type: ignore
    #         axes[k].plot(iters, jvalues, "k-", label="J = C + P")  # type: ignore
    #
    #         axes[k].set_ylabel("QoI evaluations")  # type: ignore
    #         axes[k].set_xlabel("Number of iterations")  # type: ignore
    #
    #         title = "Optimization using FOM"
    #         if k == 1:
    #             title = "Optimization using ROM"
    #         axes[k].set_title(title)  # type: ignore
    #     axes[1].legend(loc="best")  # type: ignore
    #
    #     plt.show()

    # ### Write outputs
    fom_minimization_data["method"] = args.method
    rom_minimization_data["method"] = args.method
    rom_minimization_data["num_modes"] = np.average(rec_data[0].dofs_per_vert)

    with example.fom_minimization_data.open("wb") as fh:
        dump(fom_minimization_data, fh)

    with example.rom_minimization_data(args.distr, args.name).open("wb") as fh:
        dump(rom_minimization_data, fh)


def record_results(function, parse, data, mu):
    QoI = function(mu)
    data["num_evals"] += 1
    data["evaluation_points"].append(parse(mu).to_numpy())
    data["evaluations"].append(QoI)
    print(".", end="")
    return QoI


def report(result, parse, data, reference_mu=None):

    def volume(x):
        return 10. - np.sum(np.pi * x ** 2)

    if result.status != 0:
        print("\n failed!")
    else:
        print("\n succeeded!")
        print("  mu_min:    {}".format(parse(result.x)))
        print("  J(mu_min): {}".format(result.fun))
        print("  Vol(mu_min): {}".format(volume(result.x)))
        if reference_mu is not None:
            print(
                "  absolute error in mu_min w.r.t. reference solution: {:.2e}".format(
                    np.linalg.norm(result.x - reference_mu)
                )
            )
        print("  num iterations:        {}".format(result.nit))
        print("  num function calls:    {}".format(data["num_evals"]))
        print("  time:                  {:.5f} seconds".format(data["time"]))
        if "offline_time" in data:
            print(
                "  offline time:          {:.5f} seconds".format(data["offline_time"])
            )
    print("")


def solve_optimization_problem(
    cli, initial_guess, mu_range, displacement, model, parageom, minimization_data, gradient=False, rec_data=None
):
    """solve optimization problem"""

    from functools import partial
    from scipy.optimize import minimize, NonlinearConstraint, Bounds
    from time import perf_counter
    from .stress_analysis import principal_stress_2d

    if model.name.startswith("locROM"):
        assert rec_data is not None
        dofmap, local_bases, u_local = rec_data
        from .locmor import reconstruct

        def update_displacement_solution(mu):
            U = model.solve(mu)
            reconstruct(U.to_numpy(), dofmap, local_bases, u_local, displacement)
    else:
        assert rec_data is None

        def update_displacement_solution(mu):
            U = model.solve(mu)
            displacement.x.array[:] = U.to_numpy().flatten()

    # ### Lower & upper bounds for principal Cauchy stress
    # see https://doi.org/10.1016/j.compstruc.2019.106104
    # lower_bound = -20.0 # [MPa]
    upper_bound = 2.2 # [MPa]
    confidence = cli.confidence

    def eval_objective_functional(mu):
        return model.output(mu)[0, 0]

    def active_cells(domain):
        # get active cells to deactivate constrain function
        # in some part of the domain (near the support)

        def exclude(x):
            radius = 0.3
            center = np.array([[10.0], [0.0], [0.0]])
            distance = np.linalg.norm(np.abs(x - center), axis=0)
            return distance < radius

        tdim = domain.topology.dim
        map_c = domain.topology.index_map(tdim)
        num_cells = map_c.size_local + map_c.num_ghosts
        allcells = np.arange(0, num_cells, dtype=np.int32)

        nonactive = df.mesh.locate_entities(domain, tdim, exclude)
        active = np.setdiff1d(allcells, nonactive)
        return active

    V = displacement.function_space
    # active region for the stress constraint
    constrained = active_cells(V.mesh)
    tdim = V.mesh.topology.dim
    submesh, cell_map, _, _ = df.mesh.create_submesh(V.mesh, tdim, constrained)

    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    q_degree = 2
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(submesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    def eval_rf_2(x):
        """risk factor for tension"""
        mu = model.parameters.parse(x)
        update_displacement_solution(mu)
        # U = model.solve(mu)
        # displacement.x.array[:] = U.to_numpy().flatten()
        _, s2 = principal_stress_2d(displacement, parageom, q_degree=2, cells=cell_map, values=stress.x.array.reshape(cell_map.size, -1))
        tension = s2.flatten() / upper_bound
        return confidence - tension

    lower = mu_range[0] * np.ones(10)
    upper = mu_range[1] * np.ones(10)
    bounds = Bounds(lower, upper, keep_feasible=True)


    method = cli.method
    if method == "COBYLA":
        options = {"tol": 1e-2, "catol": 1e-2}
    elif method == "COBYQA":
        options = {"f_target": 0.8}
    elif method == "SLSQP":
        options = {"ftol": 1e-4, "eps": 0.005}
    elif method == "trust-constr":
        options = {"gtol": 1e-4, "xtol": 1e-4, "barrier_tol": 1e-4, "finite_diff_rel_step": 5e-3}
    else:
        raise NotImplementedError

    constraints = ()
    if method in ("COBYLA", "SLSQP"):
        constraints = {"type": "ineq", "fun": eval_rf_2}
    elif method in ("trust-constr", "COBYQA"):
        constraints = NonlinearConstraint(eval_rf_2, 0., np.inf, keep_feasible=True)
    else:
        raise NotImplementedError

    tic = perf_counter()
    opt_result = minimize(
        partial(
            record_results,
            eval_objective_functional,
            model.parameters.parse,
            minimization_data,
        ),
        initial_guess.to_numpy(),
        method=method,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    minimization_data["time"] = perf_counter() - tic

    c2 = eval_rf_2(opt_result.x)
    strictly_positive = np.all(c2 >= 0.)
    negative_vals = c2[c2 < 0.]
    nearly_zero = np.all(np.isclose(negative_vals, 0., atol=1e-4))
    if strictly_positive:
        return opt_result
    elif nearly_zero:
        return opt_result
    else:
        raise ValueError("Constraint not satisfied by optimal solution!")


def build_localized_rom(cli, example, global_auxp, trafo_disp, parameters, ω=0.5):
    from .fom import discretize_subdomain_operators, ParaGeomLinEla
    from .ei import interpolate_subdomain_operator
    from .locmor import EISubdomainOperatorWrapper, assemble_gfem_system_with_ei
    from .dofmap_gfem import GFEMDofMap

    nreal = 0

    operator_local, rhs_local = discretize_subdomain_operators(example)
    u_local = df.fem.Function(operator_local.source.V)

    mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator_local, design="uniform", ntrain=501, modes=None, atol=0., rtol=example.mdeim_rtol, method="method_of_snapshots")
    m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
    restricted_op, _ = operator_local.restricted(m_dofs, padding=1e-8)
    wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix, magic_dofs, m_inv)

    # convert `rhs_local` to NumPy
    vector = rhs_local.as_range_array().to_numpy()
    rhs_va = mops[0].range.from_numpy(vector)
    rhs_local = VectorOperator(rhs_va)

    # ### Coarse grid of the global domain
    coarse_grid = global_auxp.coarse_grid

    # ### DofMap
    dofmap = GFEMDofMap(coarse_grid)

    # ### Reduced bases
    # 0: left, 1: transition, 2: inner, 3: transition, 4: right
    archetypes = []
    for cell in range(5):
        archetypes.append(np.load(example.local_basis_npy(nreal, cli.name, cli.distr, cell)))

    local_bases = []
    local_bases.append(archetypes[0].copy())
    local_bases.append(archetypes[1].copy())
    for _ in range(6):
        local_bases.append(archetypes[2].copy())
    local_bases.append(archetypes[3].copy())
    local_bases.append(archetypes[4].copy())
    bases_length = [len(rb) for rb in local_bases]
    
    # ### Maximum number of modes per vertex
    max_dofs_per_vert = np.load(example.local_basis_dofs_per_vert(nreal, cli.name, cli.distr))
    # raise to number of cells in the coarse grid
    repetitions = [1, 1, coarse_grid.num_cells - len(archetypes) + 1, 1, 1]
    assert np.isclose(np.sum(repetitions), coarse_grid.num_cells)
    max_dofs_per_vert = np.repeat(max_dofs_per_vert, repetitions, axis=0)
    assert max_dofs_per_vert.shape == (coarse_grid.num_cells, 4)
    assert np.allclose(np.array(bases_length), np.sum(max_dofs_per_vert, axis=1))

    Nmax = max_dofs_per_vert.max()
    ΔN = 10
    num_modes_per_vertex = list(range(Nmax // ΔN, Nmax + 1, 3 * (Nmax // ΔN) ))
    nmodes = cli.num_modes or num_modes_per_vertex[-2]

    dofs_per_vert = max_dofs_per_vert.copy()
    dofs_per_vert[max_dofs_per_vert > nmodes] = nmodes
    dofmap.distribute_dofs(dofs_per_vert)

    operator, rhs, local_bases = assemble_gfem_system_with_ei(
        dofmap, wrapped_op, rhs_local, local_bases, dofs_per_vert, max_dofs_per_vert, parameters)

    # definition of ParaGeom Problem for volume computation
    omega = global_auxp.problem.domain
    matparam = {"gdim": omega.gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(omega, global_auxp.problem.V, trafo_disp, matparam)

    def param_setter(mu):
        trafo_disp.x.array[:] = 0.0
        global_auxp.solve(trafo_disp, mu)
        trafo_disp.x.scatter_forward()

    def compute_volume(mu):
        param_setter(mu)

        vcpp = df.fem.form(parageom.form_volume)
        vol = df.fem.assemble_scalar(vcpp) # type: ignore
        return vol

    initial_mu = parameters.parse([0.1 for _ in range(example.nx)])
    vol_ref = compute_volume(initial_mu)
    U_ref = operator.apply_inverse(rhs.as_range_array(), mu=initial_mu)

    volume = GenericParameterFunctional(compute_volume, parameters)
    vol_va = NumpyVectorSpace(1).ones(1)
    vol_va.scal( (1. - ω) / vol_ref)
    one_op = ConstantOperator(vol_va, source=operator.source)

    # compliance
    compl_ref = rhs.as_range_array().inner(U_ref).item()
    scaled_fext = rhs.as_range_array().copy()
    scaled_fext.scal(1 / compl_ref)
    compliance = VectorFunctional(scaled_fext, product=None, name="compliance")

    # output J = (1 - ω) mass + ω compliance
    output = LincombOperator([one_op, compliance], [volume, ω])

    rom = StationaryModel(operator, rhs, output_functional=output, name="locROM_with_ei")
    return rom, (dofmap, local_bases, u_local)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Minimize mass (QoI) using FOM and localized ROM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "distr",
        type=str,
        help="The distribution used for sampling.",
        choices=("normal", "multivariate_normal"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument(
        "--num_modes",
        type=int,
        help="Local basis size to be used with local ROM.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("COBYLA", "COBYQA", "SLSQP", "trust-constr"),
        help="The solver to use for the minimization problem.",
        default="SLSQP",
    )
    parser.add_argument(
            "--confidence",
            type=float,
            help="Confidence (0 <= c <= 1) interval for stress constraint.",
            default=1.0
            )
    parser.add_argument(
            "--omega",
            type=float,
            help="Weighting factor for output functional.",
            default=0.5
            )
    parser.add_argument("--show", action="store_true", help="Show QoI over iterations.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
