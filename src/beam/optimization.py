"""Solution of an optimization problem"""

# check pymordemos/linear_optimization.py
# - [x] optimize with the FOM using FD
# - [x] optimize with the ROM using FD
# - [ ] optimize with the FOM using pymor gradient computation (if possible)
# - [ ] optimize with the ROM using pymor gradient computation (if possible)
from mpi4py import MPI

import numpy as np
from pymor.models.basic import StationaryModel
from pymor.core.pickle import dump


def main(args):
    """solve optimization problem for different models"""
    from .tasks import beam
    from .fom import discretize_fom

    # TODO add logger
    # TODO gather output data

    # ### Build FOM
    fom = discretize_fom(beam)

    # ### Build localized ROM
    rom = build_localized_rom(args, beam, fom.parameters)

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

    num_subdomains = beam.nx * beam.ny
    initial_guess = fom.parameters.parse([5. for _ in range(num_subdomains)])
    parameter_space = fom.parameters.space(beam.mu_range)
    bounds = [parameter_space.ranges["E"] for _ in range(num_subdomains)]

    mu_min = fom.parameters.parse([0.1 for _ in range(num_subdomains)])
    mu_max = fom.parameters.parse([10.0 for _ in range(num_subdomains)])

    # values of compliance using weights ω_i = 0.0!
    # C_mu_min = fom.output(mu_min)[0, 0] # 507.808
    # C_mu_max = fom.output(mu_max)[0, 0] # 5.07808
    J_mu_min = fom.output(mu_min)[0, 0]
    J_mu_max = fom.output(mu_max)[0, 0]
    print(f"{J_mu_min=}")
    print(f"{J_mu_max=}")

    opt_fom_result = solve_optimization_problem(
        args, initial_guess, bounds, fom, fom_minimization_data, gradient=False
    )
    mu_ref = opt_fom_result.x
    fom_minimization_data['num_iter'] = opt_fom_result.nit
    fom_minimization_data["mu_min"] = mu_ref
    fom_minimization_data["J(mu_min)"] = opt_fom_result.fun
    fom_minimization_data["status"] = opt_fom_result.status

    opt_rom_result = solve_optimization_problem(
        args, initial_guess, bounds, rom, rom_minimization_data, gradient=False
    )
    rom_minimization_data['num_iter'] = opt_rom_result.nit
    rom_minimization_data["mu_min"] = opt_rom_result.x
    rom_minimization_data["J(mu_min)"] = opt_rom_result.fun
    rom_minimization_data["status"] = opt_rom_result.status
    rom_minimization_data["abs_err_mu_min"] = np.linalg.norm(opt_rom_result.x - mu_ref)

    print("\nResult of optimization with FOM and FD")
    report(opt_fom_result, fom.parameters.parse, fom_minimization_data)

    print("\nResult of optimization with ROM and FD")
    report(
        opt_rom_result, fom.parameters.parse, rom_minimization_data, reference_mu=mu_ref
    )

    def get_prices(mus):
        f = lambda mu: np.sum( (mu - 0.1) ** 2)
        prices = []
        for mu in mus:
            prices.append(f(mu))
        return prices

    for k, data in enumerate([fom_minimization_data, rom_minimization_data]):
        jvalues = data["evaluations"]
        mus = data["evaluation_points"]
        price = get_prices(mus)
        compl = np.array(jvalues) - np.array(price)
        iters = np.arange(data["num_evals"])

        data["prices"] = price
        data["compliance"] = compl
        data["iterations"] = iters

    if args.show:
        import matplotlib.pyplot as plt

        fig = plt.figure(constrained_layout=True)
        axes = fig.subplots(nrows=2, ncols=1, sharex=True)
        
        fig.suptitle("Minimization of the objective functional J")
        for k, data in enumerate([fom_minimization_data, rom_minimization_data]):

            iters = data["iterations"]
            price = data["prices"]
            compl = data["compliance"]
            jvalues = data["evaluations"]

            axes[k].plot(iters, price, "b-", label="price P")
            axes[k].plot(iters, compl, "r-", label="compliance C")
            axes[k].plot(iters, jvalues, "k-", label="J = C + P")

            axes[k].set_ylabel("QoI evaluations")
            axes[k].set_xlabel("Number of iterations")

            title = "Optimization using FOM"
            if k == 1:
                title = "Optimization using ROM"
            axes[k].set_title(title)
        axes[1].legend(loc="best")

        plt.show()

    # ### Write outputs
    fom_minimization_data["method"] = args.method
    rom_minimization_data["method"] = args.method
    rom_minimization_data["num_modes"] = args.num_modes

    with beam.fom_minimization_data.open("wb") as fh:
        dump(fom_minimization_data, fh)

    with beam.rom_minimization_data(args.distr, args.name).open("wb") as fh:
        dump(rom_minimization_data, fh)


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
    cli, initial_guess, bounds, model, minimization_data, gradient=False
):
    """solve optimization problem"""

    from functools import partial
    from scipy.optimize import minimize
    from time import perf_counter

    def eval_objective_functional(mu):
        return model.output(mu)[0, 0]

    method = cli.method
    if method == "L-BFGS-B":
        options = {"ftol": 1e-2, "gtol": 1e-3}
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
        options=options,
    )
    minimization_data["time"] = perf_counter() - tic

    return opt_result


def build_localized_rom(cli, beam, parameters) -> StationaryModel:
    from .definitions import BeamProblem
    from .run_locrom import assemble_system

    from dolfinx.io import gmshio
    from basix.ufl import element
    from dolfinx import fem, default_scalar_type

    from pymor.basic import VectorFunctional, GenericParameterFunctional, ConstantOperator, LincombOperator, VectorOperator
    from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
    from multi.dofmap import DofMap
    from multi.domain import RectangularSubdomain
    from multi.materials import LinearElasticMaterial
    from multi.problems import LinElaSubProblem
    from multi.io import BasesLoader

    # ### Multiscale Problem Definition
    beam_problem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())

    # ### Discretize operators on single subdomain
    gdim = beam.gdim
    unit_cell_domain, _, _ = gmshio.read_from_msh(
        beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim
    )
    omega = RectangularSubdomain(12, unit_cell_domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()
    top_tag = int(137)
    omega.create_facet_tags({"top": top_tag})

    # FE space
    degree = beam.fe_deg
    fe = element("P", omega.grid.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)

    # base material
    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # Problem on unit cell domain
    problem = LinElaSubProblem(omega, V, phases=mat)
    loading = fem.Constant(
        omega.grid, (default_scalar_type(0.0), default_scalar_type(-10.0))
    )
    problem.add_neumann_bc(top_tag, loading)

    # Full operators
    problem.setup_solver()
    problem.assemble_matrix()
    problem.assemble_vector()
    A = FenicsxMatrixOperator(problem.A, V, V)
    b = VectorOperator(source.make_array([problem.b]))  # type: ignore

    # ### Read reduced bases from disk
    bases_folder = beam.bases_path(args.distr, args.name)
    num_subdomains = beam.nx * beam.ny
    bases_loader = BasesLoader(bases_folder, num_subdomains)
    bases, num_max_modes = bases_loader.read_bases()
    num_max_modes_per_cell = np.amax(num_max_modes, axis=1)
    num_min_modes_per_cell = np.amin(num_max_modes, axis=1)
    max_modes = np.amax(num_max_modes_per_cell)
    min_modes = np.amin(num_min_modes_per_cell)
    assert min_modes <= cli.num_modes
    assert cli.num_modes <= max_modes

    dofmap = DofMap(beam_problem.coarse_grid)
    operator, rhs, _ = assemble_system(cli.num_modes, dofmap, A, b, bases, num_max_modes, parameters)

    assert not rhs.parametric
    rhs_vector = rhs.as_range_array()
    compliance = VectorFunctional(rhs_vector, product=None, name="compliance")

    mu_i_min = beam.mu_range[0]
    weights = np.ones(num_subdomains)
    cost = GenericParameterFunctional(lambda mu: np.dot(weights, (mu["E"] - mu_i_min) ** 2), parameters)
    One = ConstantOperator(compliance.range.ones(1), source=compliance.source)
    objective = LincombOperator([compliance, One], [1., cost])

    rom = StationaryModel(
            operator, rhs, output_functional=objective, name="locROM"
            )
    return rom


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser(
        description="Minimize the compliance (QoI) using FOM and localized ROM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
            help="Number of fine scale modes to be used with local ROM."
            )
    parser.add_argument(
            "--method",
            type=str,
            choices=("L-BFGS-B", "SLSQP"),
            help="The solver to use for the minimization problem.",
            default="L-BFGS-B"
            )
    parser.add_argument("--show", action="store_true", help="Show QoI over iterations.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
