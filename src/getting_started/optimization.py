"""Solution of an optimization problem"""

# check pymordemos/linear_optimization.py

# - [ ] compute output functional (compliance) for chosen reference value of μ
# - [ ] plot objective function over parameter space? Not possible in my case ($R^10$). Could do this for the oversampling problem.
# - [ ] optimize with the FOM using FD
# - [ ] optimize with the ROM using FD
# - [ ] optimize with the FOM using pymor gradient computation (if possible)
# - [ ] optimize with the ROM using pymor gradient computation (if possible)
import numpy as np
from definitions import Example
from fom import discretize_fom
from pymor.core.pickle import load


def main():
    """solve optimization problem for different models"""
    beam = Example(name="beam")
    fom = discretize_fom(beam)

    with beam.reduced_model.open('rb') as fh:
        rom, parameter_space = load(fh)

    fom_minimization_data = {
            'num_evals': 0,
            'evaluations': [],
            'evaluation_points': [],
            'time': np.inf
            }


    rom_minimization_data = {
            'num_evals': 0,
            'evaluations': [],
            'evaluation_points': [],
            'time': np.inf
            }

    num_subdomains = beam.nx * beam.ny
    initial_guess = fom.parameters.parse([1.5 for _ in range(num_subdomains)])
    bounds = [parameter_space.ranges['E'] for _ in range(num_subdomains)]


    opt_fom_result = solve_optimization_problem(initial_guess, bounds, fom, fom_minimization_data, gradient=False)
    mu_ref = opt_fom_result.x


    # Note: rom.logger is a DummyLogger and therefore no output on solve
    opt_rom_result = solve_optimization_problem(initial_guess, bounds, rom, rom_minimization_data, gradient=False)

    print('\nResult of optimization with FOM and FD')
    report(opt_fom_result, fom.parameters.parse, fom_minimization_data)

    print('\nResult of optimization with ROM and FD')
    report(opt_rom_result, fom.parameters.parse, rom_minimization_data, reference_mu=mu_ref)



def record_results(function, parse, data, mu):
    QoI = function(mu)
    data['num_evals'] += 1
    data['evaluation_points'].append(parse(mu).to_numpy())
    data['evaluations'].append(QoI)
    print('.', end='')
    return QoI


def report(result, parse, data, reference_mu=None):
    if (result.status != 0):
        print('\n failed!')
    else:
        print('\n succeeded!')
        print('  mu_min:    {}'.format(parse(result.x)))
        print('  J(mu_min): {}'.format(result.fun))
        if reference_mu is not None:
            print('  absolute error in mu_min w.r.t. reference solution: {:.2e}'.format(np.linalg.norm(result.x-reference_mu)))
        print('  num iterations:        {}'.format(result.nit))
        print('  num function calls:    {}'.format(data['num_evals']))
        print('  time:                  {:.5f} seconds'.format(data['time']))
        if 'offline_time' in data:
            print('  offline time:          {:.5f} seconds'.format(data['offline_time']))
    print('')


def solve_optimization_problem(initial_guess, bounds, model, minimization_data, gradient=False):
    """solve optimization problem"""

    from functools import partial
    from scipy.optimize import minimize
    from time import perf_counter

    def eval_objective_functional(mu):
        return model.output(mu)[0, 0]

    tic = perf_counter()
    opt_result = minimize(partial(record_results, eval_objective_functional,
                                  model.parameters.parse, minimization_data),
                          initial_guess.to_numpy(),
                          method='L-BFGS-B',
                          jac=gradient,
                          bounds=bounds,
                          options={'ftol': 1e-15})
    minimization_data['time'] = perf_counter() - tic

    return opt_result


if __name__ == "__main__":
    main()
