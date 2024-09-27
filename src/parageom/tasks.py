"""Tasks for the beam example with parametrized geometry."""

import os
import shutil
from pathlib import Path

from doit.tools import run_once

from parageom.definitions import ROOT, BeamData

os.environ['PYMOR_COLORS_DISABLE'] = '1'
example = BeamData(name='scaled_parageom', debug=True)
SRC = ROOT / 'src' / 'parageom'


def rm_rf(task, dryrun):
    """Removes any target.

    If the target is a file it is removed as usual.
    If the target is a dir, it is removed even if non-empty
    in contrast to the default implementation of `doit.task.clean_targets`.
    """
    for target in sorted(task.targets, reverse=True):
        if os.path.isfile(target):
            print("%s - removing file '%s'" % (task.name, target))
            if not dryrun:
                os.remove(target)
        elif os.path.isdir(target):
            if os.listdir(target):
                msg = "%s - removing dir (although not empty) '%s'"
                print(msg % (task.name, target))
                if not dryrun:
                    shutil.rmtree(target)
            else:
                msg = "%s - removing dir '%s'"
                print(msg % (task.name, target))
                if not dryrun:
                    os.rmdir(target)


def with_h5(xdmf: Path) -> list[Path]:
    files = [xdmf, xdmf.with_suffix('.h5')]
    return files


def task_parent_unit_cell():
    """ParaGeom: Create mesh for parent unit cell."""
    from parageom.preprocessing import discretize_unit_cell

    def create_parent_unit_cell(targets):
        unit_length = example.unit_length
        mu_bar = example.parameters['subdomain'].parse([example.mu_bar])
        num_cells = example.num_intervals
        options = {'Mesh.ElementOrder': example.geom_deg}
        discretize_unit_cell(unit_length, mu_bar, num_cells, targets[0], options)

    return {
        'file_dep': [SRC / 'preprocessing.py'],
        'actions': [create_parent_unit_cell],
        'targets': [example.parent_unit_cell],
        'clean': True,
    }


def task_coarse_grid():
    """ParaGeom: Create structured coarse grids."""
    from parageom.preprocessing import create_structured_coarse_grid

    def create_global_coarse_grid(targets):
        create_structured_coarse_grid(example, 'global', targets[0])

    return {
        'file_dep': [SRC / 'preprocessing.py'],
        'actions': [create_global_coarse_grid],
        'targets': [example.coarse_grid('global')],
        'clean': True,
        'uptodate': [run_once],
    }


def task_fine_grid():
    """ParaGeom: Create parent domain mesh."""
    from parageom.preprocessing import create_fine_scale_grid

    def create_global_fine_grid(targets):
        create_fine_scale_grid(example, 'global', targets[0])

    return {
        'file_dep': [example.coarse_grid('global'), example.parent_unit_cell, SRC / 'preprocessing.py'],
        'actions': [create_global_fine_grid],
        'targets': [example.parent_domain('global')],
        'clean': True,
    }


def task_oversampling_grid():
    """ParaGeom: Create grids for oversampling."""
    source = SRC / 'preprocessing.py'
    for k in range(11):
        targets = [example.path_omega_coarse(k)]
        targets.extend(with_h5(example.path_omega(k)))
        targets.extend(with_h5(example.path_omega_in(k)))
        yield {
            'name': f'{k}',
            'file_dep': [source, example.coarse_grid('global')],
            'actions': ['python3 {} {}'.format(source, k)],
            'targets': targets,
            'clean': True,
        }


def task_preproc():
    """ParaGeom: All tasks related to preprocessing."""
    return {
        'actions': None,
        'task_dep': ['coarse_grid', 'fine_grid', 'parent_unit_cell', 'oversampling_grid'],
    }


def task_hapod():
    """ParaGeom: Construct basis via HAPOD."""

    def create_action(source, nreal, k, debug=False):
        action = f'python3 {source} {nreal} {k}'
        if debug:
            action += ' --debug'
        return action

    source = SRC / 'hapod.py'
    for nreal in range(example.num_real):
        for k in range(11):
            deps = [source]
            deps.append(example.path_omega_coarse(k))
            deps.extend(with_h5(example.path_omega(k)))
            deps.extend(with_h5(example.path_omega_in(k)))
            targets = []
            targets.append(example.log_basis_construction(nreal, 'hapod', k))
            targets.append(example.hapod_modes_npy(nreal, k))
            targets.append(example.hapod_singular_values(nreal, k))
            targets.append(example.hapod_info(nreal, k))
            if example.debug:
                targets.extend(with_h5(example.hapod_modes_xdmf(nreal, k)))
            yield {
                'name': str(nreal) + ':' + str(k),
                'file_dep': deps,
                'actions': [create_action(source, nreal, k, debug=example.debug)],
                'targets': targets,
                'clean': True,
            }


def task_hrrf():
    """ParaGeom: Construct basis via HRRF."""

    def create_action(source, nreal, k, debug=False):
        action = f'python3 {source} {nreal} {k}'
        if debug:
            action += ' --debug'
        return action

    source = SRC / 'heuristic.py'
    for nreal in range(example.num_real):
        for k in range(11):
            deps = [source]
            deps.append(example.path_omega_coarse(k))
            deps.extend(with_h5(example.path_omega(k)))
            deps.extend(with_h5(example.path_omega_in(k)))
            targets = []
            targets.append(example.log_basis_construction(nreal, 'heuristic', k))
            targets.append(example.heuristic_modes_npy(nreal, k))
            if example.debug:
                targets.extend(with_h5(example.heuristic_modes_xdmf(nreal, k)))
            yield {
                'name': str(nreal) + ':' + str(k),
                'file_dep': deps,
                'actions': [create_action(source, nreal, k, debug=example.debug)],
                'targets': targets,
                'clean': True,
            }


def task_projerr():
    """ParaGeom: Compute projection error."""
    source = SRC / 'projerr.py'
    k = 5  # use this oversampling problem
    # check sensitivity wrt mu rather than uncertainty in g
    num_samples = 100
    num_testvecs = 10
    ntrain = {'heuristic': 50, 'hapod': 200}
    ntest = {'heuristic': 200, 'hapod': None}

    def create_action_projerr(nreal, method, ntrain, output, ntest=None, debug=False):
        action = f'python3 {source} {nreal} {method} {k} {ntrain}'
        action += f' {num_samples} {num_testvecs}'
        action += f' --output {output}'
        if ntest is not None:
            action += f' --ntest {ntest}'
        if debug:
            action += ' --debug'
        return action

    for nreal in range(example.num_real):
        for method in example.methods:
            deps = [source]
            deps.append(example.path_omega_coarse(k))
            deps.extend(with_h5(example.path_omega(k)))
            deps.extend(with_h5(example.path_omega_in(k)))
            targets = []
            targets.append(example.projerr(nreal, method, k))
            targets.append(example.log_projerr(nreal, method, k))
            yield {
                'name': ':'.join([str(nreal), method, str(k)]),
                'file_dep': deps,
                'actions': [create_action_projerr(nreal, method, ntrain[method], targets[0], ntest=ntest[method])],
                'targets': targets,
                'clean': True,
            }


def task_fig_projerr():
    """ParaGeom: Plot projection error."""
    source = SRC / 'plot_projerr.py'
    k = 5
    for nreal in range(example.num_real):
        deps = [source]
        for method in example.methods:
            deps.append(example.projerr(nreal, method, k))
        targets = []
        targets.append(example.fig_projerr(k))
        yield {
            'name': ':'.join([str(nreal), str(k)]),
            'file_dep': deps,
            'actions': ['python3 {} {} {} %(targets)s'.format(source, nreal, k)],
            'targets': targets,
            'clean': True,
        }


def task_gfem():
    """ParaGeom: Build GFEM approximation."""
    source = SRC / 'gfem.py'

    def cell_to_transfer_problem(x) -> list[int]:
        r = []
        r.append(x)
        r.append(x + 1)
        return r

    def create_action(script, nreal, cell, method, debug=False):
        action = 'python3 {} {} {} {}'.format(script, nreal, cell, method)
        if debug:
            action += ' --debug'
        return action

    for nreal in range(example.num_real):
        for cell in range(10):
            for method in example.methods:
                deps = [source]
                deps.append(example.coarse_grid('global'))
                deps.append(example.parent_unit_cell)
                for k in cell_to_transfer_problem(cell):
                    deps.append(example.path_omega_in(k))
                    if method == 'hapod':
                        deps.append(example.hapod_modes_npy(nreal, k))
                    else:
                        deps.append(example.heuristic_modes_npy(nreal, k))
                targets = []
                targets.append(example.local_basis_npy(nreal, cell, method=method))
                targets.append(example.local_basis_dofs_per_vert(nreal, cell, method=method))
                targets.append(example.log_gfem(nreal, cell, method=method))
                if example.debug:
                    targets.extend(with_h5(example.local_basis_npy(nreal, cell, method=method).with_suffix('.xdmf')))
                yield {
                    'name': ':'.join([str(nreal), str(cell), method]),
                    'file_dep': deps,
                    'actions': [create_action(source, nreal, cell, method, debug=example.debug)],
                    'targets': targets,
                    'clean': True,
                }


def task_validate_rom():
    """ParaGeom: Validate ROM."""

    def create_action(source, nreal, method, num_params, num_modes, options):
        action = 'python3 {} {} {} {} {}'.format(source, nreal, method, num_params, num_modes)
        for k, v in options.items():
            action += f' {k}'
            if v:
                action += f' {v}'
        return [action]

    source = SRC / 'validate_rom.py'
    num_params = example.validate_rom['num_params']
    number_of_modes = example.validate_rom['num_modes']
    with_ei = {'ei': True}
    # with_ei = {'no_ei': False, 'ei': True}
    num_cells = example.nx * example.ny

    for nreal in range(example.num_real):
        for num_modes in number_of_modes:
            for method in example.methods:
                deps = [source]
                deps.append(example.coarse_grid('global'))
                deps.append(example.parent_domain('global'))
                deps.append(example.parent_unit_cell)
                for cell in range(num_cells):
                    deps.append(example.local_basis_npy(nreal, cell, method=method))
                    deps.append(example.local_basis_dofs_per_vert(nreal, cell, method=method))

                options = {}
                for key, value in with_ei.items():
                    targets = []
                    targets.append(example.rom_error_u(nreal, num_modes, method=method, ei=value))
                    # targets.append(example.rom_error_s(nreal, num_modes, method=method, ei=value))
                    targets.append(example.log_validate_rom(nreal, num_modes, method=method, ei=value))
                    if value:
                        options['--ei'] = ''
                    yield {
                        'name': ':'.join([str(nreal), method, str(num_modes), key]),
                        'file_dep': deps,
                        'actions': create_action(source, nreal, method, num_params, num_modes, options),
                        'targets': targets,
                        'clean': True,
                    }


def task_fig_rom_error():
    """ParaGeom: Plot ROM error."""
    source = SRC / 'plot_romerr.py'
    nreal = 0  # TODO compute mean over all realizations ...
    number_of_modes = example.validate_rom['num_modes']

    def create_action(method, output, ei=False):
        action = f'python3 {source} {nreal} {method} {output}'
        if ei:
            action += ' --ei'
        return action

    with_ei = {True: 'ei'}
    # with_ei = {False: '', True: 'ei'}
    for method in example.methods:
        for ei in [False, True]:
            deps = [source]
            for num_modes in number_of_modes:
                deps.append(example.rom_error_u(nreal, num_modes, method=method, ei=ei))
                # deps.append(example.rom_error_s(nreal, num_modes, method=method, ei=ei))
            targets = [example.fig_rom_error(method, ei=ei)]
            yield {
                'name': ':'.join([method, with_ei[ei]]),
                'file_dep': deps,
                'actions': [create_action(method, targets[0], ei=ei)],
                'targets': targets,
                'clean': True,
            }


def task_optimization():
    """ParaGeom: Determine optimal design."""
    source = SRC / 'optimization.py'

    num_modes = 100
    method = 'hapod'  # TODO add heuristic --> adjust log, targets
    minimizer = 'SLSQP'
    omega = example.omega

    nreal = 0  # do optimization only for single realization
    deps = [SRC / 'optimization.py']
    deps.append(example.coarse_grid('global'))
    deps.append(example.parent_domain('global'))
    deps.append(example.parent_unit_cell)
    for cell in range(example.nx * example.ny):
        deps.append(example.local_basis_npy(nreal, cell))
        deps.append(example.local_basis_dofs_per_vert(nreal, cell))
    targets = [example.fom_minimization_data, example.rom_minimization_data, example.log_optimization]
    return {
        'file_dep': deps,
        'actions': [
            'python3 {} {} {} --minimizer {} --omega {} --ei'.format(
                source.as_posix(), num_modes, method, minimizer, omega
            )
        ],
        'targets': targets,
        'clean': True,
    }


# def task_pp_stress():
#     """ParaGeom: Post-process stress"""
#     module = "src.parageom.pp_stress"
#     distr = "normal"
#     omega = example.omega # weighting factor for output functional
#     nreal = 0
#     for method in example.methods:
#         deps = [SRC / "pp_stress.py"]
#         # mesh and basis deps to construct rom
#         deps.append(example.coarse_grid("global"))
#         deps.append(example.parent_domain("global"))
#         deps.append(example.parent_unit_cell)
#         for cell in range(5):
#             deps.append(example.local_basis_npy(nreal, method, distr, cell))
#         deps.append(example.local_basis_dofs_per_vert(nreal, method, distr))
#         # optimization result
#         deps.append(example.fom_minimization_data)
#         # xdmf files as targets
#         targets = []
#         for x in example.pp_stress(method).values():
#             targets.extend(with_h5(x))
#         yield {
#                 "name": method,
#                 "file_dep": deps,
#                 "actions": ["python3 -m {} {} {} --omega {}".format(module, distr, method, omega)],
#                 "targets": targets,
#                 "clean": True,
#                 }
