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
    """Getting started: Build ROM"""
    return {
            "basename": f"build_rom_{ex.name}",
            "file_dep": [ROOT / "src/getting_started/rom.py"],
            "actions": ["python3 %(dependencies)s"],
            "targets": [ex.reduced_model],
            "clean": True,
            }

def task_optimize():
    """Getting started: Optimization"""
    optpy = ROOT / "src/getting_started/optimization.py"
    return {
            "basename": f"optimize_{ex.name}",
            "file_dep": [optpy, ex.reduced_model],
            "actions": ["python3 {}".format(optpy)],
            "verbosity": 2,
            "uptodate": [False],
            }
