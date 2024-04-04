"""tasks for the beam example with parametrized geometry"""

import os
from .definitions import BeamData, ROOT

os.environ["PYMOR_COLORS_DISABLE"] = "1"
example = BeamData(name="parageom")
SRC = ROOT / "src" / f"{example.name}"


def task_parent_unit_cell():
    """ParaGeom: Create mesh for parent unit cell"""
    from .preprocessing import discretize_unit_cell

    def create_parent_unit_cell(targets):
        mu_bar = example.parameters.parse([0.2])
        num_cells = example.num_intervals
        discretize_unit_cell(mu_bar, num_cells, targets[0])

    return {
            "file_dep": [SRC / "preprocessing.py"],
            "actions": [create_parent_unit_cell],
            "targets": [example.parent_unit_cell],
            "clean": True,
            }
