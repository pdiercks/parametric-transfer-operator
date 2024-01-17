"""tasks for the getting started example"""

from .definitions import Example, ROOT
from pathlib import Path
from doit.tools import run_once

SRC = ROOT / "src/getting_started"  # source for this example
defs = SRC / "definitions.py"
beam = Example(name="beam")


def with_h5(xdmf: Path) -> list[Path]:
    files = [xdmf, xdmf.with_suffix(".h5")]
    return files


def task_preprocessing():
    """Getting started: Preprocessing"""
    from .preprocessing import generate_meshes

    return {
        "basename": f"preproc_{beam.name}",
        "file_dep": [defs, SRC / "preprocessing.py"],
        "actions": [(generate_meshes, [beam])],
        "targets": [
            beam.coarse_grid,
            beam.unit_cell_grid,
            *with_h5(beam.fine_grid),
            beam.coarse_oversampling_grid,
            *with_h5(beam.fine_oversampling_grid),
        ],
        "clean": True,
        "uptodate": [run_once],
    }


def task_build_rom():
    """Getting started: Build ROM"""
    return {
        "basename": f"build_rom_{beam.name}",
        "file_dep": [
            beam.coarse_grid,
            beam.unit_cell_grid,
            beam.fine_grid,
            defs,
            SRC / "fom.py",
            SRC / "rom.py",
        ],
        "actions": ["python3 -m src.getting_started.rom"],
        "targets": [beam.reduced_model, beam.singular_values],
        "clean": True,
    }


def task_loc_pod_modes():
    """Getting started: Construct local POD basis"""
    module = "src.getting_started.range_approximation"
    file = SRC / "range_approximation.py"
    distributions = ("normal", "multivariate_normal")
    num_train = 10  # number of Transfer operators
    nworkers = 4  # number of workers in pool
    for distr in distributions:
        yield {
            "basename": f"rrf_{beam.name}_{distr}",
            "file_dep": [
                defs,
                file,
                beam.fine_oversampling_grid,
                beam.coarse_oversampling_grid,
            ],
            "actions": [
                "python3 -m {} {} {} --max_workers {}".format(
                    module, distr, num_train, nworkers
                )
            ],
            "targets": [
                beam.range_approximation_log(distr),
                beam.loc_pod_modes(distr),
                beam.loc_singular_values(distr),
            ],
            "clean": True,
        }


def task_test_sets():
    """Getting started: Generate FOM test sets"""
    module = "src.getting_started.fom_test_set"
    code = SRC / "fom_test_set.py"
    num_solves = 10
    for subdomain in (4,):
        yield {
            "basename": f"test_set_{subdomain}_{beam.name}",
            "file_dep": [
                defs,
                code,
                beam.coarse_grid,
                beam.unit_cell_grid,
                *with_h5(beam.fine_grid),
            ],
            "actions": ["python3 -m {} {} {}".format(module, num_solves, subdomain)],
            "targets": [],
            "clean": True,
        }


def task_decomposition():
    """Getting started: Decompose local POD basis"""
    module = "src.getting_started.decompose_pod_basis"
    code = SRC / "decompose_pod_basis.py"
    distributions = ("normal", "multivariate_normal")
    for distr in distributions:
        yield {
            "basename": f"decompose_{beam.name}_{distr}",
            "file_dep": [defs, code, beam.unit_cell_grid, beam.loc_pod_modes(distr)],
            "actions": ["python3 -m {} {}".format(module, distr)],
            "targets": [
                beam.local_basis_npz(distr),
                *with_h5(beam.fine_scale_modes_xdmf(distr)),
                *with_h5(beam.pod_modes_xdmf(distr)),
            ],
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
