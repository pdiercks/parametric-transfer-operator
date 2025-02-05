"""Tasks for the beam example with parametrized geometry."""

import os
import shutil
from pathlib import Path

from doit.tools import run_once

from parageom.definitions import ROOT, BeamData

os.environ['PYMOR_COLORS_DISABLE'] = '1'
example = BeamData(name='parageom-validation', debug=False)
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
        unit_length = example.preproc.unit_length
        num_cells = example.preproc.num_intervals
        options = {'Mesh.ElementOrder': example.preproc.geom_deg}
        discretize_unit_cell(unit_length, example.mu_bar, num_cells, targets[0], options)

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
        create_structured_coarse_grid(example, targets[0])

    return {
        'file_dep': [SRC / 'preprocessing.py'],
        'actions': [create_global_coarse_grid],
        'targets': [example.coarse_grid],
        'clean': True,
        'uptodate': [run_once],
    }


def task_fine_grid():
    """ParaGeom: Create parent domain mesh."""
    from parageom.preprocessing import create_fine_scale_grid

    def create_global_fine_grid(targets):
        create_fine_scale_grid(example, targets[0])

    return {
        'file_dep': [example.coarse_grid, example.parent_unit_cell, SRC / 'preprocessing.py'],
        'actions': [create_global_fine_grid],
        'targets': [example.fine_grid],
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
            'file_dep': [source, example.coarse_grid],
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
            targets.append(example.modes_npy('hapod', nreal, k))
            targets.append(example.hapod_singular_values(nreal, k))
            targets.append(example.hapod_summary(nreal, k))
            if example.debug:
                targets.extend(with_h5(example.modes_xdmf('hapod', nreal, k)))
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
            targets.append(example.log_basis_construction(nreal, 'hrrf', k))
            targets.append(example.modes_npy('hrrf', nreal, k))
            if example.debug:
                targets.extend(with_h5(example.modes_xdmf('hrrf', nreal, k)))
            yield {
                'name': str(nreal) + ':' + str(k),
                'file_dep': deps,
                'actions': [create_action(source, nreal, k, debug=example.debug)],
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
                deps.append(example.coarse_grid)
                deps.append(example.parent_unit_cell)
                for k in cell_to_transfer_problem(cell):
                    deps.append(example.path_omega_in(k))
                    deps.append(example.modes_npy(method, nreal, k))
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


# def task_projerr():
#     """ParaGeom: Compute projection error."""
#     source = SRC / 'projerr.py'
#     num_samples = 50
#     # check sensitivity wrt mu rather than uncertainty in g
#     num_testvecs = 10
#     N = 400
#     ntrain_hrrf = {'hrrf': 50, 'hapod': None}
#     amplitudes = [example.g_scale]
#
#     def create_action_projerr(nreal, method, k, ntrain, output, ntrain_hrrf=None, scale=None, debug=False):
#         action = f'python3 {source} {nreal} {method} {k} {ntrain}'
#         action += f' {num_samples} {num_testvecs}'
#         action += f' --output {output}'
#         if ntrain_hrrf is not None:
#             action += f' --ntrain_hrrf {ntrain_hrrf}'
#         if scale is not None:
#             action += f' --scale {scale}'
#         if debug:
#             action += ' --debug'
#         return action
#
#     for nreal in range(example.num_real):
#         for k in example.projerr.configs:
#             for method in example.methods:
#                 deps = [source]
#                 deps.append(example.path_omega_coarse(k))
#                 deps.extend(with_h5(example.path_omega(k)))
#                 deps.extend(with_h5(example.path_omega_in(k)))
#                 for scale in amplitudes:
#                     targets = []
#                     targets.append(example.projection_error(nreal, method, k, scale))
#                     targets.append(example.log_projerr(nreal, method, k, scale))
#                     yield {
#                         'name': ':'.join([str(nreal), method, str(k), str(scale)]),
#                         'file_dep': deps,
#                         'actions': [
#                             create_action_projerr(
#                                 nreal, method, k, N, targets[0], ntrain_hrrf=ntrain_hrrf[method], scale=scale
#                             )
#                         ],
#                         'targets': targets,
#                         'clean': True,
#                     }


# def task_pp_projerr():
#     """ParaGeom: Postprocess projection error."""
#     from parageom.postprocessing import compute_mean_std
#
#     for method in example.methods:
#         for k in example.projerr.configs:
#             # gather data
#             deps = []
#             for n in range(example.num_real):
#                 deps.append(example.projection_error(n, method, k))
#             yield {
#                 'name': ':'.join([method, str(k)]),
#                 'file_dep': deps,
#                 'actions': [compute_mean_std],
#                 'targets': [example.mean_projection_error(method, k)],
#                 'clean': True,
#             }


# def task_fig_projerr():
#     """ParaGeom: Plot projection error."""
#     source = SRC / 'plot_projerr.py'
#     scale = example.g_scale
#     for k in example.projerr.configs:
#         deps = [source]
#         deps.append(example.mean_projection_error('hapod', k))
#         deps.append(example.mean_projection_error('hrrf', k))
#         targets = [example.fig_projerr(k)]
#         yield {
#             'name': ':'.join([str(k)]),
#             'file_dep': deps,
#             'actions': ['python3 {} {} {} %(targets)s'.format(source, k, scale)],
#             'targets': targets,
#             'clean': True,
#         }


# def task_fig_max_projerr():
#     """ParaGeom: Plot max projection error."""
#     source = SRC / 'plot_max_projerr.py'
#     scale = example.g_scale
#     for k in example.projerr.configs:
#         deps = [source]
#         deps.append(example.mean_projection_error('hapod', k))
#         deps.append(example.mean_projection_error('hrrf', k))
#         targets = [example.fig_max_projerr(k)]
#         yield {
#             'name': ':'.join([str(k)]),
#             'file_dep': deps,
#             'actions': ['python3 {} {} {} %(targets)s'.format(source, k, scale)],
#             'targets': targets,
#             'clean': True,
#         }


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
    num_params = example.rom_validation.ntest
    number_of_modes = example.rom_validation.num_modes
    with_ei = {'ei': True}
    # with_ei = {'no_ei': False, 'ei': True}
    num_cells = example.nx * example.ny

    for nreal in range(example.num_real):
        for num_modes in number_of_modes:
            for method in example.methods:
                deps = [source]
                deps.append(example.coarse_grid)
                deps.append(example.fine_grid)
                deps.append(example.parent_unit_cell)
                for cell in range(num_cells):
                    deps.append(example.local_basis_npy(nreal, cell, method=method))
                    deps.append(example.local_basis_dofs_per_vert(nreal, cell, method=method))

                options = {}
                for key, value in with_ei.items():
                    targets = []
                    for field in example.rom_validation.fields:
                        targets.append(example.rom_error(method, nreal, field, num_modes, ei=value))
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


def task_pp_rom_error():
    """ParaGeom: Postprocess ROM error."""

    def gather_and_compute(example, method, field, targets):
        """Gather error data for all number of modes, then compute mean & std."""
        import numpy as np

        Nmodes = len(example.rom_validation.num_modes)

        output = {}
        errkeys = {
            'u': ['energy_min', 'energy_avg', 'energy_max', 'max_min', 'max_avg', 'max_max'],
            's': ['euclidean_min', 'euclidean_avg', 'euclidean_max', 'max_min', 'max_avg', 'max_max'],
        }
        for key in errkeys[field]:
            error = []
            # collect error (min/avg/max over validation set) for each realization
            for n in range(example.num_real):
                err = []
                # grow error over number of modes
                for num_modes in example.rom_validation.num_modes:
                    infile = example.rom_error(method, n, field, num_modes, ei=True)
                    data = np.load(infile)
                    err.append(data[key])
                error.append(np.array(err))
            error = np.vstack(error)
            assert error.shape == (example.num_real, Nmodes)
            output[f'mean_{key}'] = np.mean(error, axis=0)
            output[f'std_{key}'] = np.std(error, axis=0)
        np.savez(targets[0], **output)

    fields = example.rom_validation.fields
    for method in example.methods:
        for qty in fields:
            # gather data
            deps = []
            for n in range(example.num_real):
                for nmodes in example.rom_validation.num_modes:
                    deps.append(example.rom_error(method, n, qty, nmodes, ei=True))
            yield {
                'name': ':'.join([method, qty]),
                'file_dep': deps,
                'actions': [(gather_and_compute, [example, method, qty])],
                'targets': [example.mean_rom_error(method, qty, ei=True)],
                'clean': True,
            }


def task_fig_rom_error():
    """ParaGeom: Plot ROM error."""
    source = SRC / 'plot_romerr.py'

    def create_action(field, output, ei=False):
        action = f'python3 {source} {field} {output}'
        if ei:
            action += ' --ei'
        return action

    with_ei = {True: 'ei'}
    for field in example.rom_validation.fields:
        for ei in with_ei.keys():
            deps = [source]
            for method in example.methods:
                deps.append(example.mean_rom_error(method, field, ei=ei))
            targets = [example.fig_rom_error(field, ei=ei)]
            yield {
                'name': ':'.join([field, with_ei[ei]]),
                'file_dep': deps,
                'actions': [create_action(field, targets[0], ei=ei)],
                'targets': targets,
                'clean': True,
            }


def task_optimization():
    """ParaGeom: Determine optimal design."""
    source = SRC / 'optimization.py'

    nreal = example.opt.nreal
    omega = example.opt.omega
    num_modes = example.opt.num_modes
    method = example.opt.method
    minimizer = example.opt.minimizer

    deps = [source]
    deps.append(example.coarse_grid)
    deps.append(example.fine_grid)
    deps.append(example.parent_unit_cell)
    for cell in range(example.nx * example.ny):
        deps.append(example.local_basis_npy(nreal, cell, method=method))
        deps.append(example.local_basis_dofs_per_vert(nreal, cell, method=method))
    targets = [
        example.fom_minimization_data(method, nreal),
        example.rom_minimization_data(method, nreal),
        example.log_optimization,
    ]
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


def task_pp_hrrf_basis_length():
    """ParaGeom: Average HRRF Modes."""

    def determine_average(targets):
        from collections import defaultdict

        import numpy as np

        from parageom.postprocessing import parse_logfile

        method = 'hrrf'
        basis_length = defaultdict(list)
        for k in range(11):
            search = {k: f'Final basis length (k={k:02d}):'}

            for n in range(example.num_real):
                logfile = example.log_basis_construction(n, method, k)
                data = parse_logfile(logfile.as_posix(), search)
                basis_length[k].extend(data[k])

        # average basis length for each configuration
        average = np.zeros(11)
        for key, array in basis_length.items():
            average[key] = np.mean(array)

        np.save(targets[0], average)

    return {
        'file_dep': [example.log_basis_construction(n, 'hrrf', k) for k in range(11) for n in range(example.num_real)],
        'actions': [determine_average],
        'targets': [example.method_folder('hrrf') / 'mean_basis_length.npy'],
        'clean': True,
    }


def task_pp_hapod_basis_length():
    """ParaGeom: Average HAPOD Modes."""

    def determine_average(targets):
        import numpy as np

        from parageom.postprocessing import average_hapod_data

        data = average_hapod_data(example)
        np.savez(targets[0], **data)

    return {
        'file_dep': [example.hapod_summary(n, k) for n in range(example.num_real) for k in range(11)],
        'actions': [determine_average],
        'targets': [example.method_folder('hapod') / 'mean_basis_length.npz'],
        'clean': True,
    }


def task_fig_basis_size():
    """ParaGeom: Average basis size."""
    source = SRC / 'plot_mean_basis_length.py'
    hrrf = example.method_folder('hrrf') / 'mean_basis_length.npy'
    hapod = example.method_folder('hapod') / 'mean_basis_length.npz'
    cmdaction = 'python3 {} {} {} %(targets)s'.format(source, hapod, hrrf)
    return {
        'file_dep': [source, hapod, hrrf],
        'actions': [cmdaction],
        'targets': [example.fig_basis_size()],
        'clean': True,
    }


def task_mdeim_data():
    """ParaGeom: Write MDEIM data."""
    source = SRC / 'ei.py'
    return {
        'file_dep': [source],
        'actions': ['python %(dependencies)s'],
        'targets': [example.mdeim_data()],
        'clean': True,
    }


def task_fig_mdeim_svals():
    """ParaGeom: Figure MDEIM Svals."""
    source = SRC / 'plot_svals_simple.py'
    return {
        'file_dep': [example.mdeim_data()],
        'actions': ['python3 {} %(dependencies)s %(targets)s'.format(source)],
        'targets': [example.fig_mdeim_svals()],
        'clean': True,
    }


def task_fig_aux_svals():
    """ParaGeom: Figure AUX Svals."""
    source = SRC / 'plot_svals_simple.py'
    return {
        'file_dep': [example.singular_values_auxiliary_problem],
        'actions': ['python3 {} %(dependencies)s %(targets)s'.format(source)],
        'targets': [example.fig_aux_svals()],
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
