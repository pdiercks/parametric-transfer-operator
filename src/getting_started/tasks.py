"""tasks for the getting started example"""

from .definitions import Example, ROOT
from doit.tools import run_once
SRC = ROOT / "src/getting_started"  # source for this example

ex = Example(name="beam")


def task_preprocessing():
    """Getting started: Preprocessing"""
    from .preprocessing import generate_meshes
    return {
            "basename": f"preproc_{ex.name}",
            "file_dep": [SRC / "definitions.py"],
            "actions": [(generate_meshes, [ex])],
            "targets": [ex.coarse_grid, ex.unit_cell_grid, ex.fine_grid, ex.coarse_oversampling_grid, ex.fine_oversampling_grid],
            "clean": True,
            "uptodate": [run_once],
            }


def task_build_rom():
    """Getting started: Build ROM"""
    return {
            "basename": f"build_rom_{ex.name}",
            "file_dep": [ex.coarse_grid, ex.unit_cell_grid, ex.fine_grid, SRC/ "definitions.py", SRC / "fom.py", SRC / "rom.py"],
            "actions": ["python3 {}".format(SRC / "rom.py")],
            "targets": [ex.reduced_model, ex.singular_values],
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
