"""tasks for the beam example with parametrized geometry"""

import os
import shutil
from pathlib import Path
from doit.tools import run_once
from .definitions import BeamData, ROOT

os.environ["PYMOR_COLORS_DISABLE"] = "1"
example = BeamData(name="parageom")
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


def task_training_sets():
    """ParaGeom: Create training sets"""
    from .lhs import sample_lhs
    import numpy as np
    from pymor.parameters.base import ParameterSpace
    from pymor.core.pickle import dump

    def create_training_set(config, seed, targets):
        parameter_space = ParameterSpace(example.parameters[config], example.mu_range)
        name = list(example.parameters[config].keys())[0]
        num_samples = example.ntrain(config)
        train = sample_lhs(
            parameter_space,
            name=name,
            samples=num_samples,
            criterion="center",
            random_state=seed,
        )
        with open(targets[0], "wb") as fh:
            dump({"training_set": train, "seed": seed}, fh)

    seed = example.training_set_seed
    random_seeds = np.random.SeedSequence(seed).generate_state(len(CONFIGS))

    for config, seed in zip(CONFIGS, random_seeds):
        yield {
            "name": config,
            "file_dep": [],
            "actions": [(create_training_set, [config, seed])],
            "targets": [example.training_set(config)],
            "clean": True,
            "uptodate": [run_once],
        }


def task_coarse_grid():
    """ParaGeom: Create structured coarse grids"""
    module = "src.parageom.preprocessing"

    for config in list(CONFIGS) + ["global"]:
        yield {
                "name": config,
                "file_dep": [],
                "actions": ["python3 -m {} {} {} --output %(targets)s".format(module, config, "coarse")],
                "targets": [example.coarse_grid(config)],
                "clean": True,
                "uptodate": [run_once],
                }


def task_global_parent_domain():
    """ParaGeom: Create global parent domain mesh"""
    module = "src.parageom.preprocessing"
    config = "global"
    return {
            "file_dep": [example.coarse_grid(config)],
            "actions": ["python3 -m {} {} {} --output %(targets)s".format(module, config, "parent")],
            "targets": [example.global_parent_domain],
            "clean": True,
            }


def task_oversampling_grids():
    """ParaGeom: Create physical meshes (Ω, Ω_in) for each μ"""
    module = "src.parageom.preprocessing"

    for config in CONFIGS:
        ntrain = example.ntrain(config)
        targets = []
        for k in range(ntrain):
            targets.extend(with_h5(example.oversampling_domain(config, k)))
            targets.extend(with_h5(example.target_subdomain(config, k)))
        yield {
                "name": config,
                "file_dep": [example.parent_unit_cell, example.training_set(config), example.coarse_grid(config)],
                "actions": ["python3 -m {} {} {}".format(module, config, "oversampling")],
                "targets": targets,
                "clean": True,
                }


def task_preproc():
    """ParaGeom: All tasks related to preprocessing"""
    return {
            "actions": None,
            "task_dep": ["oversampling_grids", "global_parent_domain", "coarse_grid", "training_sets", "parent_unit_cell"]
            }


def task_hapod():
    """ParaGeom: Construct edge basis via HAPOD"""
    module = "src.parageom.hapod"
    nworkers = 4  # number of workers in pool
    distr = example.distributions[0]
    for nreal in range(example.num_real):
        for config in CONFIGS:
            deps = [SRC / "hapod.py"]
            # global grids needed for BeamProblem initialization
            deps.append(example.coarse_grid("global"))
            deps.append(example.global_parent_domain)
            targets = []
            for k in range(example.ntrain(config)):
                deps.append(example.oversampling_domain(config, k))
                targets.extend(with_h5(
                        example.hapod_snapshots(nreal, distr, config, k)
                        ))
            targets.append(example.log_edge_basis(nreal, "hapod", distr, config))
            targets.append(example.rrf_bases_length(nreal, "hapod", distr, config))
            targets.append(example.fine_scale_edge_modes_npz(nreal, "hapod", distr, config))
            targets.append(example.hapod_singular_values_npz(nreal, distr, config))
            targets.append(example.hapod_pod_data(nreal, distr, config))
            yield {
                "name": config+":"+str(nreal),
                "file_dep": deps,
                "actions": [
                    "python3 -m {} {} {} {} --max_workers {}".format(
                        module, distr, config, nreal, nworkers
                    )
                ],
                "targets": targets,
                "clean": True,
            }


def task_extension():
    """ParaGeom: Extend fine scale modes and write final basis"""
    module = "src.parageom.extension"
    num_cells = example.nx * example.ny
    distr = example.distributions[0]
    for nreal in range(example.num_real):
        for method in example.methods:
            for cell_index in range(num_cells):
                config = example.cell_to_config(cell_index)
                yield {
                    "name": f"{method}_{cell_index}",
                    "file_dep": [example.fine_scale_edge_modes_npz(nreal, method, distr, config)],
                    "actions": [
                        "python3 -m {} {} {} {} {}".format(module, nreal, method, distr, cell_index)
                    ],
                    "targets": [
                        example.local_basis_npz(nreal, method, distr, cell_index),
                        example.fine_scale_modes_bp(nreal, method, distr, cell_index),
                        example.log_extension(nreal, method, distr, cell_index),
                    ],
                    "clean": [rm_rf],
                }
