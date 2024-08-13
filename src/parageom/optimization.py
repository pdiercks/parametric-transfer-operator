"""Solution of an optimization problem"""

# NOTE
# the below code was written based on the pyMOR Tutorial
# "Model order reduction for PDE-constrained optimization problems"
# See https://docs.pymor.org/2023-2-0/tutorial_optimization.html

import dolfinx as df
import basix

import numpy as np
from multi.materials import LinearElasticMaterial
from pymor.models.basic import StationaryModel
from pymor.core.pickle import dump


def main(args):
    """solve optimization problem for different models"""
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, ParaGeomLinEla

    # ### Build FOM
    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    degree = example.geom_deg
    interface_tags = [
        i for i in range(15, 25)
    ]
    auxiliary_problem = discretize_auxiliary_problem(
        parent_domain_path,
        degree,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    V = auxiliary_problem.problem.V
    d_trafo = df.fem.Function(V, name="d_trafo")
    fom = discretize_fom(example, auxiliary_problem, d_trafo)

    # ### Build localized ROM
    # rom = build_localized_rom(args, beam, fom.parameters)

    # ### Global Functions for displacement and Cauchy stress
    displacement = df.fem.Function(V, name="u")
    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    q_degree = 2
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(V.mesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    # after U <-- model.solve(mu)
    # displacement is updated with values of U independent of which model was used
    # and the stress analysis is carried out over the full mesh

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

    num_subdomains = example.nx * example.ny
    initial_guess = fom.parameters.parse([0.2 for _ in range(num_subdomains)])
    parameter_space = fom.parameters.space(example.mu_range)
    bounds = [parameter_space.ranges["R"] for _ in range(num_subdomains)]
    # ### Stress constraints
    # (C1) s_min < s_1 < s_max
    # (C2) s_min < s_2 < s_max
    # RF1: max( s_1 / s_max, s_1 / s_min)
    # RF2: max( s_2 / s_max, s_2 / s_min)
    # RF1 or RF2 not equal to 1 means that (C1), (C2) respectively, are satisfied
    # The risk factor constraint can thus be posed as inequality constraint
    # (COBYLA only supports inequality constraints)
    # okay: RF1 <= 1, RF2 <= 1, failure: RF_i > 1
    # constraint function result is expected to be non-negative, r > 0
    # constraint_rf_1 = lambda x: 1 - RF1

    # Constraint definition moved to `solve_optimization_problem` ...
    gdim = fom.solution_space.V.mesh.geometry.dim
    matparam = {"gdim": gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(auxiliary_problem.problem.domain, V, d_trafo, matparam)

    # ### Solve optimization problem using FOM
    opt_fom_result = solve_optimization_problem(
        args, initial_guess, bounds, displacement, stress, fom, parageom, fom_minimization_data, gradient=False
    )
    mu_ref = opt_fom_result.x
    fom_minimization_data["num_iter"] = opt_fom_result.nit
    fom_minimization_data["mu_min"] = mu_ref
    fom_minimization_data["J(mu_min)"] = opt_fom_result.fun
    fom_minimization_data["status"] = opt_fom_result.status

    # ### Solve Optimization Problem using ROM
    # opt_rom_result = solve_optimization_problem(
    #     args, initial_guess, bounds, rom, rom_minimization_data, gradient=False
    # )
    # rom_minimization_data["num_iter"] = opt_rom_result.nit
    # rom_minimization_data["mu_N_min"] = opt_rom_result.x
    # rom_minimization_data["J_N(mu_N_min)"] = opt_rom_result.fun
    # rom_minimization_data["status"] = opt_rom_result.status
    # rom_minimization_data["abs_err_mu"] = np.linalg.norm(opt_rom_result.x - mu_ref)
    # rom_minimization_data["abs_err_J"] = abs(opt_rom_result.fun - opt_fom_result.fun)

    # FOM output evaluated at mu_min found by ROM
    # mu_N_min = fom.parameters.parse(opt_rom_result.x)
    # J_mu_N_min = fom.output(mu_N_min)[0, 0] # type: ignore
    # rom_minimization_data["J(mu_N_min)"] = J_mu_N_min
    # rom_minimization_data["suboptimality"] = abs(J_mu_N_min - opt_fom_result.fun) / opt_fom_result.fun

    print("\nResult of optimization with FOM and FD")
    report(opt_fom_result, fom.parameters.parse, fom_minimization_data)

    # print("\nResult of optimization with ROM and FD")
    # report(
    #     opt_rom_result, fom.parameters.parse, rom_minimization_data, reference_mu=mu_ref
    # )

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
    rom_minimization_data["num_modes"] = args.num_modes

    with example.fom_minimization_data.open("wb") as fh:
        dump(fom_minimization_data, fh)

    # with example.rom_minimization_data(args.distr, args.name).open("wb") as fh:
    #     dump(rom_minimization_data, fh)


def record_results(function, parse, data, mu):
    QoI = function(mu)
    data["num_evals"] += 1
    data["evaluation_points"].append(parse(mu).to_numpy())
    data["evaluations"].append(QoI)
    print(".", end="")
    return QoI


def report(result, parse, data, reference_mu=None):
    if result.status != 0:
        print("\n failed!")
    else:
        print("\n succeeded!")
        print("  mu_min:    {}".format(parse(result.x)))
        print("  J(mu_min): {}".format(result.fun))
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
    cli, initial_guess, bounds, displacement, stress, model, parageom, minimization_data, gradient=False
):
    """solve optimization problem"""

    from functools import partial
    from scipy.optimize import minimize
    from time import perf_counter
    from .stress_analysis import principal_stress_2d

    # ### Lower & upper bounds for principal Cauchy stress
    # see https://doi.org/10.1016/j.compstruc.2019.106104
    lower_bound = -20.0 # [MPa]
    upper_bound = 2.2 # [MPa]

    mesh = displacement.function_space.mesh
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    def eval_objective_functional(mu):
        return model.output(mu)[0, 0]

    def eval_rf_1(x):
        mu = model.parameters.parse(x)
        U = model.solve(mu) # retrieve from cache or compute?
        displacement.x.array[:] = U.to_numpy().flatten()
        s1, _ = principal_stress_2d(displacement, parageom, q_degree=2, values=stress.x.array.reshape(cells.size, -1))
        compression = np.max(s1 / lower_bound)
        tension = np.max(s1 / upper_bound)
        return 1. - max(compression, tension)

    def eval_rf_2(x):
        mu = model.parameters.parse(x)
        U = model.solve(mu) # retrieve from cache or compute?
        displacement.x.array[:] = U.to_numpy().flatten()
        _, s2 = principal_stress_2d(displacement, parageom, q_degree=2, values=stress.x.array.reshape(cells.size, -1))
        compression = np.max(s2 / lower_bound)
        tension = np.max(s2 / upper_bound)
        return 1. - max(compression, tension)

    constraints = (
            {"type": "ineq", "fun": eval_rf_1},
            {"type": "ineq", "fun": eval_rf_2}
            )

    method = cli.method
    if method == "COBYLA":
        options = {"tol": 1e-2, "catol": 1e-2}
    elif method == "SLSQP":
        options = {"ftol": 1e-2}
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

    breakpoint()
    print("check constraint for optimal solution")

    return opt_result


def build_localized_rom(cli, beam, parameters) -> StationaryModel:
    raise NotImplementedError


if __name__ == "__main__":
    import sys, argparse

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
        "num_modes",
        type=int,
        help="Local basis size to be used with local ROM.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("COBYLA", "SLSQP"),
        help="The solver to use for the minimization problem.",
        default="SLSQP",
    )
    parser.add_argument("--show", action="store_true", help="Show QoI over iterations.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
