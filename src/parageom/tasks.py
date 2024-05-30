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

    for config in list(CONFIGS) + ["global"]:
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


def task_parent_domain():
    """ParaGeom: Create parent domain mesh"""
    module = "src.parageom.preprocessing"
    for config in list(CONFIGS) + ["global"]:
        yield {
            "name": config,
            "file_dep": [example.coarse_grid(config), example.parent_unit_cell],
            "actions": [
                "python3 -m {} {} {} --output %(targets)s".format(
                    module, config, "parent"
                )
            ],
            "targets": [example.parent_domain(config)],
            "clean": True,
        }


def task_preproc():
    """ParaGeom: All tasks related to preprocessing"""
    return {
        "actions": None,
        "task_dep": ["parent_domain", "coarse_grid", "parent_unit_cell"],
    }


def task_hapod():
    """ParaGeom: Construct basis via HAPOD"""
    module = "src.parageom.hapod"
    nworkers = 4  # number of workers in pool
    distr = example.distributions[0]
    for nreal in range(example.num_real):
        for config in CONFIGS:
            deps = [SRC / "hapod.py"]
            # global grids needed for BeamProblem initialization
            deps.append(example.coarse_grid("global"))
            deps.append(example.parent_domain("global"))
            deps.append(example.coarse_grid(config))
            deps.append(example.parent_domain(config))
            targets = []
            targets.append(
                example.log_basis_construction(nreal, "hapod", distr, config)
            )
            targets.append(example.hapod_modes_npy(nreal, distr, config))
            targets.extend(with_h5(example.hapod_modes_xdmf(nreal, distr, config)))
            targets.append(example.hapod_singular_values(nreal, distr, config))
            yield {
                "name": config + ":" + str(nreal),
                "file_dep": deps,
                "actions": [
                    "python3 -m {} {} {} {} --max_workers {}".format(
                        module, distr, config, nreal, nworkers
                    )
                ],
                "targets": targets,
                "clean": True,
            }


def task_heuristic():
    """ParaGeom: Construct basis via Heuristic range finder"""
    module = "src.parageom.heuristic"
    distr = example.distributions[0]
    for nreal in range(example.num_real):
        for config in CONFIGS:
            deps = [SRC / "heuristic.py"]
            # global grids needed for BeamProblem initialization
            deps.append(example.coarse_grid("global"))
            deps.append(example.parent_domain("global"))
            deps.append(example.coarse_grid(config))
            deps.append(example.parent_domain(config))
            targets = []
            targets.append(
                example.log_basis_construction(nreal, "heuristic", distr, config)
            )
            targets.append(example.heuristic_modes_npy(nreal, distr, config))
            targets.extend(with_h5(example.heuristic_modes_xdmf(nreal, distr, config)))
            yield {
                "name": config + ":" + str(nreal),
                "file_dep": deps,
                "actions": [
                    "python3 -m {} {} {} {}".format(module, distr, config, nreal)
                ],
                "targets": targets,
                "clean": True,
            }


def task_projerr():
    """ParaGeom: Compute projection error"""
    module = "src.parageom.projerr"
    distr = example.distributions[0]
    for nreal in range(example.num_real):
        for method in example.methods:
            for config in CONFIGS:
                deps = [SRC / "projerr.py"]
                deps.append(example.coarse_grid("global"))
                deps.append(example.parent_domain("global"))
                deps.append(example.parent_unit_cell)
                deps.append(example.hapod_modes_npy(nreal, distr, config))
                deps.append(example.heuristic_modes_npy(nreal, distr, config))
                targets = []
                targets.append(example.projerr(nreal, method, distr, config))
                yield {
                        "name": method + ":" + config + ":" + str(nreal),
                        "file_dep": deps,
                        "actions": ["python3 -m {} {} {} {} --output %(targets)s".format(module, method, distr, config)],
                        "targets": targets,
                        "clean": True
                        }
