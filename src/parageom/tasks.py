"""Tasks for the beam example with parametrized geometry."""

import os
import shutil
from pathlib import Path

from doit.tools import run_once

from parageom.definitions import ROOT, BeamData

os.environ['PYMOR_COLORS_DISABLE'] = '1'
example = BeamData(name='parageom', debug=True)
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


def task_projerr():
    """ParaGeom: Compute projection error."""
    source = SRC / 'projerr.py'
    num_samples = 100
    # check sensitivity wrt mu rather than uncertainty in g
    num_testvecs = 1
    # TODO: show plots for different N in the thesis?
    N = 200
    ntrain_hrrf = {'hrrf': 50, 'hapod': None}
    amplitudes = [example.g_scale]

    def create_action_projerr(nreal, method, k, ntrain, output, ntrain_hrrf=None, scale=None, debug=False):
        action = f'python3 {source} {nreal} {method} {k} {ntrain}'
        action += f' {num_samples} {num_testvecs}'
        action += f' --output {output}'
        if ntrain_hrrf is not None:
            action += f' --ntrain_hrrf {ntrain_hrrf}'
        if scale is not None:
            action += f' --scale {scale}'
        if debug:
            action += ' --debug'
        return action

    for nreal in range(example.num_real):
        for k in example.projerr.configs:
            for method in example.methods:
                deps = [source]
                deps.append(example.path_omega_coarse(k))
                deps.extend(with_h5(example.path_omega(k)))
                deps.extend(with_h5(example.path_omega_in(k)))
                for scale in amplitudes:
                    targets = []
                    targets.append(example.projection_error(nreal, method, k, scale))
                    targets.append(example.log_projerr(nreal, method, k, scale))
                    yield {
                        'name': ':'.join([str(nreal), method, str(k), str(scale)]),
                        'file_dep': deps,
                        'actions': [
                            create_action_projerr(
                                nreal, method, k, N, targets[0], ntrain_hrrf=ntrain_hrrf[method], scale=scale
                            )
                        ],
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


def task_pp_projerr():
    """ParaGeom: Postprocess projection error."""
    from parageom.postprocessing import compute_mean_std

    for method in example.methods:
        for k in example.projerr.configs:
            # gather data
            deps = []
            for n in range(example.num_real):
                deps.append(example.projection_error(n, method, k))
            yield {
                'name': ':'.join([method, str(k)]),
                'file_dep': deps,
                'actions': [(compute_mean_std, ['min', 'avg', 'max'])],
                'targets': [example.mean_projection_error(method, k)],
            }


def task_fig_projerr():
    """ParaGeom: Plot projection error."""
    source = SRC / 'plot_projerr.py'
    amplitudes = [example.g_scale]
    for nreal in range(example.num_real):
        for k in example.projerr.configs:
            deps = [source]
            for scale in amplitudes:
                for method in example.methods:
                    deps.append(example.projection_error(nreal, method, k, scale))
                targets = []
                targets.append(example.fig_projerr(k, scale))
                yield {
                    'name': ':'.join([str(nreal), str(k), str(scale)]),
                    'file_dep': deps,
                    'actions': ['python3 {} {} {} {} %(targets)s'.format(source, nreal, k, scale)],
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
            }


def task_fig_rom_error():
    """ParaGeom: Plot ROM error."""
    source = SRC / 'plot_romerr.py'

    # TODO: (A) plot min, avg, max over realizations for max error over validation set
    # TODO: (B) plot min, avg, max over validation set for single realization

    def create_action(method, output, ei=False):
        action = f'python3 {source} {method} {output}'
        if ei:
            action += ' --ei'
        return action

    with_ei = {True: 'ei'}
    # with_ei = {False: '', True: 'ei'}
    for method in example.methods:
        for ei in with_ei.keys():
            deps = [source]
            deps.append(example.mean_rom_error(method, 'u', ei=ei))
            deps.append(example.mean_rom_error(method, 's', ei=ei))
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
