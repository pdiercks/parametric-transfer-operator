"""Solution of an optimization problem."""

# NOTE
# the below code was written based on the pyMOR Tutorial
# "Model order reduction for PDE-constrained optimization problems"
# See https://docs.pymor.org/2023-2-0/tutorial_optimization.html

import pathlib
from collections import namedtuple

import basix
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from multi.dofmap import DofMap
from multi.domain import StructuredQuadGrid
from multi.interpolation import make_mapping
from multi.io import read_mesh
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.core.pickle import dump
from pymor.parameters.base import Mu
from scipy.optimize import Bounds, NonlinearConstraint, minimize

from parageom.auxiliary_problem import AuxiliaryModelWrapper

StressWrapper = namedtuple('StressWrapper', ['expr', 'cells', 'fun'])
# TODO Re-evaluate design of FOM and ROM (and ModelWrapper)
ModelWrapper = namedtuple(
    'ModelWrapper', ['model', 'solution', 'dofmap', 'modes', 'sol_local', 'aux', 'trafo'], defaults=(None,) * 5
)


def reconstruct(
    U_rb: np.ndarray,
    mu: Mu,
    dofmap: DofMap,
    bases: list[np.ndarray],
    Vsub: df.fem.FunctionSpace,
    u_global: df.fem.Function,
    d_global: df.fem.Function,
    aux: AuxiliaryModelWrapper,
) -> None:
    """Reconstructs ROM displacement solution & transformation displacement on the global domain.

    Args:
        U_rb: ROM solution in the reduced space.
        mu: The current parameter value.
        dofmap: The dofmap of the reduced space.
        bases: Local basis for each subdomain.
        Vsub: The local FE space.
        u_global: The global solution field to be filled with values.
        d_global: The global transformation displacement field to be filled with values.
        aux: The model of the local auxiliary problem.

    """
    coarse_grid = dofmap.grid
    V = u_global.function_space
    submesh = Vsub.mesh
    x_submesh = submesh.geometry.x

    u_global_view = u_global.x.array
    u_global_view[:] = 0.0
    d_global_view = d_global.x.array
    d_global_view[:] = 0.0

    mu_values = mu.to_numpy()
    rom = aux.model
    reductor = aux.reductor

    for i, cell in enumerate(range(dofmap.num_cells)):
        # translate subdomain mesh
        vertices = coarse_grid.get_entities(0, cell)
        dx_cell = coarse_grid.get_entity_coordinates(0, vertices)[0]
        x_submesh += dx_cell

        # fill u_local with rom solution
        basis = bases[cell]
        dofs = dofmap.cell_dofs(cell)

        # fill global field via dof mapping
        V_to_Vsub = make_mapping(Vsub, V, padding=1e-8, check=True)
        u_global_view[V_to_Vsub] = U_rb[0, dofs] @ basis

        mu_i = rom.parameters.parse(mu_values[i])
        drb = rom.solve(mu_i)
        d_global_view[V_to_Vsub] = reductor.reconstruct(drb).to_numpy()[0, :]

        # move subdomain mesh to origin
        x_submesh -= dx_cell

    u_global.x.scatter_forward()
    d_global.x.scatter_forward()


def main(args):
    """Solve optimization problem for different models."""
    from parageom.dofmap_gfem import GFEMDofMap
    from parageom.fom import ParaGeomLinEla
    from parageom.tasks import example
    from parageom.validate_rom import build_fom, build_rom

    stem = pathlib.Path(__file__).stem
    # always use first realization for optimization, nreal=0
    logfilename = example.log_optimization.as_posix()
    set_defaults({'pymor.core.logger.getLogger.filename': logfilename})
    if args.debug:
        loglevel = 10  # debug
    else:
        loglevel = 20  # info
    logger = getLogger(stem, level=loglevel)

    # ### Build FOM
    logger.info('Start building FOM ...')
    fom, parageom_fom = build_fom(example, ω=args.omega)

    # ### Function for displacement on unit cell (for reconstruction)
    unit_cell_domain = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={'gdim': example.gdim})[0]
    V_i = df.fem.functionspace(unit_cell_domain, ('P', example.fe_deg, (example.gdim,)))
    u_local = df.fem.Function(V_i, name='u_i')
    # d_local = df.fem.Function(V_i, name='d_i')

    # ### Build localized ROM
    coarse_grid_path = example.coarse_grid('global')
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': 2})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)
    dofmap = GFEMDofMap(struct_grid)
    params = fom.parameters
    logger.info('Start building ROM ...')
    rom, selected_modes, aux = build_rom(
        example, dofmap, params, args.num_modes, ω=args.omega, nreal=0, method=args.method, use_ei=True
    )

    # ### Global Function for displacement solution
    V = fom.solution_space.V
    u_fom = df.fem.Function(V, name='u_fom')
    u_rom = df.fem.Function(V, name='u_rom')
    d_rom = df.fem.Function(V, name='d_rom')  # global transformation displacement

    def constrained_cells(domain):
        """Get active cells to deactivate constraint function in some part of the domain (near the support)."""

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

    # active region for the stress constraint
    constrained = constrained_cells(V.mesh)
    submesh, cell_map, _, _ = df.mesh.create_submesh(V.mesh, V.mesh.topology.dim, constrained)

    def compute_principal_components(f, num_cells):
        values = f.reshape(num_cells, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx + fyy) / 2 - np.sqrt(((fxx - fyy) / 2) ** 2 + fxy**2)
        fmax = (fxx + fyy) / 2 + np.sqrt(((fxx - fyy) / 2) ** 2 + fxy**2)
        return fmin, fmax

    # ### Quadrature space for stress
    # Note that the function is defined on the submesh, such that fun.x.array has correct size
    # The reduced evaluation is achieved by passing cells=cell_map
    basix_celltype = getattr(basix.CellType, submesh.topology.cell_type.name)
    q_degree = 2
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    qve = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme='default', degree=q_degree)  # type: ignore
    QV = df.fem.functionspace(submesh, qve)

    stress_fom = df.fem.Function(QV)
    stress_rom = df.fem.Function(QV)

    # ### UFL representation and Expression of stress for both models
    suf = parageom_fom.weighted_stress(u_fom)
    stress_ufl_fom_vector = ufl.as_vector([suf[0, 0], suf[1, 1], suf[2, 2], suf[0, 1]])
    stress_expr_fom = df.fem.Expression(stress_ufl_fom_vector, q_points)

    # FIXME: use individual instance of ParaGeom for ROM
    rommat = {
        'gdim': parageom_fom.domain.gdim,
        'E': example.youngs_modulus,
        'NU': example.poisson_ratio,
        'plane_stress': example.plane_stress,
    }
    parageom_rom = ParaGeomLinEla(parageom_fom.domain, V, d_rom, rommat)
    sur = parageom_rom.weighted_stress(u_rom)
    stress_ufl_rom_vector = ufl.as_vector([sur[0, 0], sur[1, 1], sur[2, 2], sur[0, 1]])
    stress_expr_rom = df.fem.Expression(stress_ufl_rom_vector, q_points)

    # ### Finally wrap stress objects
    SigmaFom = StressWrapper(stress_expr_fom, cell_map, stress_fom)
    SigmaRom = StressWrapper(stress_expr_rom, cell_map, stress_rom)

    # ### Constraint function per model
    def evaluate_constraint(wrapped_model, mu, stress_expr, cells, stress_fun):
        model = wrapped_model.model
        if model.name == 'FOM':
            # in the case of the FOM, d_trafo is updated before solve
            U = model.solve(mu)
            wrapped_model.solution.x.array[:] = U.to_numpy().flatten()
        else:
            urb = model.solve(mu)
            # reconstruct updates global displacement solution and
            # transformation displacement
            reconstruct(
                urb.to_numpy(),
                mu,
                wrapped_model.dofmap,
                wrapped_model.modes,
                wrapped_model.sol_local.function_space,
                wrapped_model.solution,
                wrapped_model.trafo,
                wrapped_model.aux,
            )

        # now that u & d are uptodate, compute stress
        values = stress_fun.x.array.reshape(cells.size, -1)
        stress_expr.eval(V.mesh, entities=cells, values=values)
        s1, s2 = compute_principal_components(values, cells.size)
        return s1, s2

    # ### Dict's to gather minimization data
    fom_minimization_data = {
        'num_evals': 0,
        'evaluations': [],
        'evaluation_points': [],
        'time': np.inf,
    }

    rom_minimization_data = {
        'num_evals': 0,
        'evaluations': [],
        'evaluation_points': [],
        'time': np.inf,
    }

    # ### Solve optimization problem using FOM
    num_subdomains = example.nx * example.ny
    initial_guess = fom.parameters.parse([0.1 for _ in range(num_subdomains)])
    parameter_space = fom.parameters.space(example.mu_range)
    mu_range = parameter_space.ranges['R']

    lower = mu_range[0] * np.ones(num_subdomains)
    upper = mu_range[1] * np.ones(num_subdomains)
    bounds = Bounds(lower, upper, keep_feasible=True)

    # Wrap models
    wrapped_fom = ModelWrapper(fom, u_fom)
    wrapped_rom = ModelWrapper(rom, u_rom, dofmap, selected_modes, u_local, aux, d_rom)

    # ### Solve Optimization Problem using FOM
    opt_fom_result = solve_optimization_problem(
        logger,
        args,
        initial_guess,
        bounds,
        evaluate_constraint,
        wrapped_fom,
        SigmaFom,
        fom_minimization_data,
        gradient=False,
    )
    mu_ref = opt_fom_result.x
    fom_minimization_data['num_iter'] = opt_fom_result.nit
    fom_minimization_data['mu_min'] = mu_ref
    fom_minimization_data['J(mu_min)'] = opt_fom_result.fun
    fom_minimization_data['status'] = opt_fom_result.status

    # ### Solve Optimization Problem using ROM
    opt_rom_result = solve_optimization_problem(
        logger,
        args,
        initial_guess,
        bounds,
        evaluate_constraint,
        wrapped_rom,
        SigmaRom,
        rom_minimization_data,
        gradient=False,
    )
    rom_minimization_data['num_iter'] = opt_rom_result.nit
    rom_minimization_data['mu_N_min'] = opt_rom_result.x
    rom_minimization_data['J_N(mu_N_min)'] = opt_rom_result.fun
    rom_minimization_data['status'] = opt_rom_result.status
    rom_minimization_data['abs_err_mu'] = np.linalg.norm(opt_rom_result.x - mu_ref)
    rom_minimization_data['abs_err_J'] = abs(opt_rom_result.fun - opt_fom_result.fun)

    # FOM output evaluated at mu_min found by ROM
    mu_N_min = fom.parameters.parse(opt_rom_result.x)
    J_mu_N_min = fom.output(mu_N_min)[0, 0]  # type: ignore
    rom_minimization_data['J(mu_N_min)'] = J_mu_N_min
    rom_minimization_data['suboptimality'] = abs(J_mu_N_min - opt_fom_result.fun) / opt_fom_result.fun

    report(logger, fom.name, opt_fom_result, fom.parameters.parse, fom_minimization_data)
    report(logger, rom.name, opt_rom_result, fom.parameters.parse, rom_minimization_data, reference_mu=mu_ref)

    # TODO: what data should be shown in the paper?

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
    fom_minimization_data['method'] = args.minimizer
    rom_minimization_data['method'] = args.minimizer
    rom_minimization_data['num_modes'] = args.num_modes

    with example.fom_minimization_data.open('wb') as fh:
        dump(fom_minimization_data, fh)

    with example.rom_minimization_data.open('wb') as fh:
        dump(rom_minimization_data, fh)


def record_results(function, parse, data, mu):
    QoI = function(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(parse(mu).to_numpy())
    data['evaluations'].append(QoI)
    return QoI


def report(logger, model_name, result, parse, data, reference_mu=None):
    def volume(x):
        return 10.0 - np.sum(np.pi * x**2)

    if result.status != 0:
        status = 'failed!'
    else:
        status = 'succeded!'

    if reference_mu is not None:
        absolute_error = np.linalg.norm(result.x - reference_mu)
    else:
        absolute_error = 0.0

    if 'offline_time' in data:
        offline_time = data['offline_time']
    else:
        offline_time = 0.0

    summary = f"""\nResult of optimization with {model_name} and FD

    status:        {status}
    mu_min:        {parse(result.x)}
    J(mu_min):     {result.fun}
    Vol(mu_min):   {volume(result.x)}
    num iterations:        {result.nit}
    num function calls:    {data['num_evals']}
    time:                  {data['time']:.5f} seconds

    offline time:          {offline_time:.5f} seconds
    absolute error in mu_min w.r.t. reference solution:  {absolute_error:.2e}
    """

    logger.info(summary)


def solve_optimization_problem(
    logger, cli, initial_guess, bounds, compute_stress, wrapped_model, stress, minimization_data, gradient=False
):
    """Solve optimization problem."""
    from functools import partial
    from time import perf_counter

    # ### Lower & upper bounds for principal Cauchy stress
    # see https://doi.org/10.1016/j.compstruc.2019.106104
    upper_bound = 2.2  # [MPa]
    confidence = cli.confidence
    model = wrapped_model.model

    def eval_objective_functional(mu):
        return model.output(mu)[0, 0]

    def eval_rf_2(x):
        """Risk factor for tension."""
        mu = model.parameters.parse(x)
        _, s2 = compute_stress(wrapped_model, mu, stress.expr, stress.cells, stress.fun)
        tension = s2.flatten() / upper_bound
        return confidence - tension

    minimizer = cli.minimizer
    if minimizer == 'COBYLA':
        options = {'tol': 1e-2, 'catol': 1e-2}
    elif minimizer == 'COBYQA':
        options = {'f_target': 0.8}
    elif minimizer == 'SLSQP':
        options = {'ftol': 1e-3, 'eps': 0.01}
    elif minimizer == 'trust-constr':
        options = {'gtol': 1e-4, 'xtol': 1e-4, 'barrier_tol': 1e-4, 'finite_diff_rel_step': 5e-3}
    else:
        raise NotImplementedError

    constraints = ()
    if minimizer in ('COBYLA', 'SLSQP'):
        constraints = {'type': 'ineq', 'fun': eval_rf_2}
    elif minimizer in ('trust-constr', 'COBYQA'):
        constraints = NonlinearConstraint(eval_rf_2, 0.0, np.inf, keep_feasible=True)
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
        method=minimizer,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options=options,
    )
    minimization_data['time'] = perf_counter() - tic

    c2 = eval_rf_2(opt_result.x)
    strictly_positive = np.all(c2 >= 0.0)
    if strictly_positive:
        return opt_result
    else:
        negative_vals = c2[c2 < 0.0]
        assert negative_vals.size > 0
        min_value = np.amin(negative_vals)
        logger.warning(
            f'Constraint not satisfied by optimal solution (max constraint violation is {np.abs(min_value):1.2e}!'
        )
        return opt_result


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Minimize Output Functional using FOM and localized ROM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'num_modes',
        type=int,
        help='Local basis size to be used with local ROM.',
    )
    parser.add_argument('method', type=str, help='Method used to generate local bases.')
    parser.add_argument(
        '--minimizer',
        type=str,
        choices=('COBYLA', 'COBYQA', 'SLSQP', 'trust-constr'),
        help='The solver to use for the minimization problem.',
        default='SLSQP',
    )
    parser.add_argument(
        '--confidence', type=float, help='Confidence (0 <= c <= 1) interval for stress constraint.', default=1.0
    )
    parser.add_argument('--omega', type=float, help='Weighting factor for output functional.', default=0.5)
    parser.add_argument('--show', action='store_true', help='Show QoI over iterations.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--ei', action='store_true', help='Use ROM with EI.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
