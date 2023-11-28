"""tasks for the getting started example"""

from .definitions import Example
from doit.tools import run_once

ex = Example(name="beam")


def task_preprocessing():
    """preprocessing: getting started"""
    from .preprocessing import generate_meshes
    return {
            "basename": f"preproc_{ex.name}",
            "actions": [(generate_meshes, [ex])],
            "targets": [ex.coarse_grid, ex.unit_cell_grid, ex.fine_grid],
            "clean": True,
            "uptodate": [run_once],
            }
