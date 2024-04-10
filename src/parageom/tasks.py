"""tasks for the beam example with parametrized geometry"""

import os
from .definitions import BeamData, ROOT

os.environ["PYMOR_COLORS_DISABLE"] = "1"
example = BeamData(name="parageom")
SRC = ROOT / "src" / f"{example.name}"
CONFIGS = example.configurations


def task_parent_unit_cell():
    """ParaGeom: Create mesh for parent unit cell"""
    from .preprocessing import discretize_unit_cell

    def create_parent_unit_cell(targets):
        mu_bar = example.parameters["subdomain"].parse([0.2])
        num_cells = example.num_intervals
        discretize_unit_cell(mu_bar, num_cells, targets[0])

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
        # for left and right I will get the same meshes actually
        # but this should not be a problem
        with open(targets[0], "wb") as fh:
            dump({"training_set": train, "seed": seed}, fh)

    # use realization to generate seed
    # FIXME at the moment there is only one realization
    realizations = np.load(example.realizations)
    random_seeds = np.random.SeedSequence(realizations[0]).generate_state(len(CONFIGS))

    for config, seed in zip(CONFIGS, random_seeds):
        yield {
            "name": config,
            "file_dep": [example.parent_unit_cell],
            "actions": [(create_training_set, [config, seed])],
            "targets": [example.training_set(config)],
            "clean": True,
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
                }


def task_oversampling_grids():
    """ParaGeom: Create physical mesh for each μ"""
    module = "src.parageom.preprocessing"

    for config in CONFIGS:
        ntrain = example.ntrain(config)
        targets = []
        for k in range(ntrain):
            targets.append(example.oversampling_domain(config, k))
        yield {
                "name": config,
                "file_dep": [example.training_set(config), example.coarse_grid(config)],
                "actions": ["python3 -m {} {} {}".format(module, config, "oversampling")],
                "targets": targets,
                "clean": True,
                }
