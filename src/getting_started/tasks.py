"""tasks for the getting started example"""

from .definitions import Example, ROOT
from doit.tools import run_once

ex = Example(name="beam")


def task_preprocessing():
    """Getting started: Preprocessing"""
    from .preprocessing import generate_meshes
    return {
            "basename": f"preproc_{ex.name}",
            "actions": [(generate_meshes, [ex])],
            "targets": [ex.coarse_grid, ex.unit_cell_grid, ex.fine_grid],
            "clean": True,
            "uptodate": [run_once],
            }


def task_build_rom():
    """Getting starded: Build ROM"""
    return {
            "basename": f"build_rom_{ex.name}",
            "file_dep": [ROOT / "src/getting_started/rom.py"],
            "actions": ["python3 %(dependencies)s"],
            "targets": [ex.reduced_model],
            "clean": True,
            }
