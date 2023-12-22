"""tasks for the getting started example"""

from .definitions import Example, ROOT
from doit.tools import run_once
SRC = ROOT / "src/getting_started"  # source for this example
beam = Example(name="beam")


def task_preprocessing():
    """Getting started: Preprocessing"""
    from .preprocessing import generate_meshes
    return {
            "basename": f"preproc_{beam.name}",
            "file_dep": [SRC / "definitions.py"],
            "actions": [(generate_meshes, [beam])],
            "targets": [beam.coarse_grid, beam.unit_cell_grid, beam.fine_grid, beam.coarse_oversampling_grid, beam.fine_oversampling_grid],
            "clean": True,
            "uptodate": [run_once],
            }


def task_build_rom():
    """Getting started: Build ROM"""
    return {
            "basename": f"build_rom_{beam.name}",
            "file_dep": [beam.coarse_grid, beam.unit_cell_grid, beam.fine_grid, SRC/ "definitions.py", SRC / "fom.py", SRC / "rom.py"],
            "actions": ["python3 -m src.getting_started.rom"],
            "targets": [beam.reduced_model, beam.singular_values],
            "clean": True,
            }


def task_optimize():
    """Getting started: Optimization"""
    optpy = ROOT / "src/getting_started/optimization.py"
    return {
            "basename": f"optimize_{beam.name}",
            "file_dep": [optpy, beam.reduced_model],
            "actions": ["python3 -m src.getting_started.optimization"],
            "verbosity": 2,
            "uptodate": [False],
            }
