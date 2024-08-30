"""tasks for the beam example with parametrized geometry"""

import os
import shutil
from pathlib import Path
from doit.tools import run_once
from .definitions import BeamData, ROOT

os.environ["PYMOR_COLORS_DISABLE"] = "1"
example = BeamData(name="parageom", run_mode="DEBUG")
SRC = ROOT / "src" / f"{example.name}"
CONFIGS = example.configurations


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
    files = [xdmf, xdmf.with_suffix(".h5")]
    return files


def task_parent_unit_cell():
    """ParaGeom: Create mesh for parent unit cell"""
    from .preprocessing import discretize_unit_cell

    def create_parent_unit_cell(targets):
        unit_length = example.unit_length
        mu_bar = example.parameters["subdomain"].parse([example.mu_bar])
        num_cells = example.num_intervals
        options = {"Mesh.ElementOrder": example.geom_deg}
        discretize_unit_cell(unit_length, mu_bar, num_cells, targets[0], options)

    return {
        "file_dep": [SRC / "preprocessing.py"],
        "actions": [create_parent_unit_cell],
        "targets": [example.parent_unit_cell],
        "clean": True,
    }


def task_coarse_grid():
    """ParaGeom: Create structured coarse grids"""
    module = "src.parageom.preprocessing"

    for config in ["global"]:
        yield {
            "name": config,
            "file_dep": [],
            "actions": [
                "python3 -m {} {} {} --output %(targets)s".format(
                    module, config, "coarse"
                )
            ],
            "targets": [example.coarse_grid(config)],
            "clean": True,
            "uptodate": [run_once],
        }


def task_fine_grid():
    """ParaGeom: Create parent domain mesh"""
    module = "src.parageom.preprocessing"
    for config in ["global"]:
        yield {
            "name": config,
            "file_dep": [example.coarse_grid(config), example.parent_unit_cell],
            "actions": [
                "python3 -m {} {} {} --output %(targets)s".format(
                    module, config, "fine"
                )
            ],
            "targets": [example.parent_domain(config)],
            "clean": True,
        }


def task_preproc():
    """ParaGeom: All tasks related to preprocessing"""
    return {
        "actions": None,
        "task_dep": ["coarse_grid", "fine_grid", "parent_unit_cell"],
    }


def task_hapod():
    """ParaGeom: Construct basis via HAPOD"""
    module = "src.parageom.osp_v2"

    for nreal in range(example.num_real):
        for k in range(11):
            deps = [SRC / "osp_v2.py"]
            deps.append(example.coarse_grid("global"))
            targets = []
            targets.append(
                example.log_basis_construction(nreal, "hapod", k)
            )
            targets.append(example.path_omega_coarse(k))
            targets.append(example.path_omega(k))
            targets.append(example.path_omega_in(k))
            targets.append(example.hapod_modes_npy(nreal, k))
            targets.append(example.hapod_singular_values(nreal, k))
            targets.extend(with_h5(example.hapod_modes_xdmf(nreal, k)))
            yield {
                "name": str(nreal) + ":" + str(k),
                "file_dep": deps,
                "actions": [
                    "python3 -m {} {} {} --debug".format(module, nreal, k)
                ],
                "targets": targets,
                "clean": True,
            }


# def task_projerr():
#     """ParaGeom: Compute projection error"""
#     module = "src.parageom.projerr"
#     distr = example.distributions[0]
#     for nreal in range(example.num_real):
#         for method in example.methods:
#             for config in CONFIGS:
#                 deps = [SRC / "projerr.py"]
#                 deps.append(example.coarse_grid("global"))
#                 deps.append(example.parent_domain("global"))
#                 deps.append(example.coarse_grid("target"))
#                 deps.append(example.parent_domain("target"))
#                 deps.append(example.coarse_grid(config))
#                 deps.append(example.parent_domain(config))
#                 deps.append(example.parent_unit_cell)
#                 targets = []
#                 targets.append(example.projerr(nreal, method, distr, config))
#                 targets.append(example.log_projerr(nreal, method, distr, config))
#                 yield {
#                         "name": method + ":" + config + ":" + str(nreal),
#                         "file_dep": deps,
#                         "actions": ["python3 -m {} {} {} {} {} --output {}".format(module, nreal, method, distr, config, targets[0])],
#                         "targets": targets,
#                         "clean": True
#                         }


# def task_fig_projerr():
#     """ParaGeom: Plot projection error"""
#     module = "src.parageom.plot_projerr"
#     distr = example.distributions[0]
#     for nreal in range(example.num_real):
#         for config in CONFIGS:
#             deps = [SRC / "plot_projerr.py"]
#             for method in example.methods:
#                 deps.append(example.projerr(nreal, method, distr, config))
#             targets = []
#             targets.append(example.fig_projerr(config))
#             yield {
#                     "name": config + ":" + str(nreal),
#                     "file_dep": deps,
#                     "actions": ["python3 -m {} {} {} %(targets)s".format(module, nreal, config)],
#                     "targets": targets,
#                     "clean": True,
#                     }


def task_gfem():
    """ParaGeom: Build GFEM approximation"""
    module = "src.parageom.gfem"

    def cell_to_transfer_problem(x) -> list[int]:
        r = []
        r.append(x)
        r.append(x + 1)
        return r

    def create_action(module, nreal, cell, debug=False):
        action = "python3 -m {} {} {}".format(module, nreal, cell)
        if debug:
            action += " --debug"
        return action

    for nreal in range(example.num_real):
        for cell in range(10):
            deps = [SRC / "gfem.py"]
            # TODO: add meshes as deps
            deps.append(example.coarse_grid("global"))
            deps.append(example.parent_unit_cell)
            for k in cell_to_transfer_problem(cell):
                deps.append(example.path_omega_in(k))
                deps.append(example.hapod_modes_npy(nreal, k))
            targets = []
            targets.append(example.local_basis_npy(nreal, cell))
            targets.append(example.local_basis_dofs_per_vert(nreal, cell))
            targets.append(example.log_gfem(nreal, cell))
            yield {
                    "name": str(nreal) + ":" + str(cell),
                    "file_dep": deps,
                    "actions": [create_action(module, nreal, cell, debug=True)],
                    "targets": targets,
                    "clean": True,
                    }


def task_validate_rom():
    """ParaGeom: Validate ROM"""

    def create_action(nreal, num_params, num_modes, options):
        action = "python3 -m src.parageom.validate_rom {} {} {}".format(nreal, num_params, num_modes)
        for k, v in options.items():
            action += f" {k}"
            if v:
                action += f" {v}"
        return [action]

    num_params = 200
    number_of_modes = [20, 40, 60, 80, 100]
    with_ei = {"no_ei": False, "ei": True}
    num_cells = example.nx * example.ny

    for nreal in range(example.num_real):
        for num_modes in number_of_modes:
            deps = [SRC / "validate_rom.py"]
            deps.append(example.coarse_grid("global"))
            deps.append(example.parent_domain("global"))
            deps.append(example.parent_unit_cell)
            for cell in range(num_cells):
                deps.append(example.local_basis_npy(nreal, cell))
                deps.append(example.local_basis_dofs_per_vert(nreal, cell))

            options = {}
            for k, v in with_ei.items():
                targets = []
                targets.append(example.rom_error_u(nreal, num_modes, ei=v))
                targets.append(example.rom_error_s(nreal, num_modes, ei=v))
                if v:
                    options["--ei"] = ""
                yield {
                        "name": ":".join([str(nreal), str(num_modes), k]),
                        "file_dep": deps,
                        "actions": create_action(nreal, num_params, num_modes, options),
                        "targets": targets,
                        "clean": True,
                        }


# def task_fig_locrom_error():
#     """ParaGeom: Plot ROM error"""
#     module = "src.parageom.plot_romerr"
#     nreal = 0
#     distr = "normal"
#     norm = "max_relerr_h1_semi"
#
#     deps = []
#     for method in example.methods:
#         deps.append(example.locrom_error(nreal, method, distr, ei=False))
#         deps.append(example.locrom_error(nreal, method, distr, ei=True))
#     return {
#             "file_dep": deps,
#             "actions": ["python3 -m {} {} --norm {} --ei %(targets)s".format(module, nreal, norm)],
#             "targets": [example.fig_locrom_error],
#             "clean": True,
#             }



# def task_optimization():
#     """ParaGeom: Determine optimal design"""
#     module = "src.parageom.optimization"
#     distr = example.distributions[0]
#
#     num_modes = 120
#     minimizer = "SLSQP"
#     omega = example.omega
#
#     nreal = 0 # do optimization only for single realization
#     for method in example.methods:
#         deps = [SRC / "optimization.py"]
#         deps.append(example.coarse_grid("global"))
#         deps.append(example.parent_domain("global"))
#         deps.append(example.parent_unit_cell)
#         for cell in range(5):
#             deps.append(example.local_basis_npy(nreal, method, distr, cell))
#         deps.append(example.local_basis_dofs_per_vert(nreal, method, distr))
#         targets = [example.fom_minimization_data,
#                    example.rom_minimization_data(distr, method)]
#         yield {
#                 "name": method,
#                 "file_dep": deps,
#                 "actions": ["python3 -m {} {} {} --num_modes {} --method {} --omega {}".format(module, distr, method, num_modes, minimizer, omega)],
#                 "targets": targets,
#                 "clean": True,
#                 }


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
