"""tasks for the getting started example"""

from .definitions import BeamData, ROOT
from pathlib import Path
from doit.tools import run_once

SRC = ROOT / "src/getting_started"  # source for this example
defs = SRC / "definitions.py"
beam = BeamData(name="beam")
CONFIGS = beam.configurations
DISTR = beam.distributions


def with_h5(xdmf: Path) -> list[Path]:
    files = [xdmf, xdmf.with_suffix(".h5")]
    return files


def task_preprocessing():
    """Getting started: Preprocessing"""
    from .preprocessing import generate_meshes

    mesh_files = [beam.coarse_grid, beam.unit_cell_grid,
                  *with_h5(beam.fine_grid)]
    for config in ("inner", "left", "right"):
        mesh_files += with_h5(beam.fine_oversampling_grid(config))

    return {
        "basename": f"preproc_{beam.name}",
        "file_dep": [defs, SRC / "preprocessing.py"],
        "actions": [(generate_meshes, [beam])],
        "targets": mesh_files,
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
    num_train = 10  # number of Transfer operators
    nworkers = 4  # number of workers in pool
    for distr in DISTR:
        for config in CONFIGS:
            yield {
                "basename": f"rrf_{beam.name}_{distr}_{config}",
                "file_dep": [
                    defs,
                    file,
                    beam.fine_oversampling_grid(config),
                ],
                "actions": [
                    "python3 -m {} {} {} {} --max_workers {}".format(
                        module, distr, num_train, config, nworkers
                    )
                ],
                "targets": [
                    beam.range_approximation_log(distr, config),
                    beam.loc_pod_modes(distr, config),
                    beam.loc_singular_values(distr, config),
                ],
                "clean": True,
            }


def task_test_sets():
    """Getting started: Generate FOM test sets"""
    module = "src.getting_started.fom_test_set"
    code = SRC / "fom_test_set.py"
    num_solves = 10
    map = {"inner": 4, "left": 0, "right": 9}
    for config in CONFIGS:
        subdomain = map[config]
        yield {
            "basename": f"test_set_{config}_{beam.name}",
            "file_dep": [
                defs,
                code,
                beam.coarse_grid,
                beam.unit_cell_grid,
                *with_h5(beam.fine_grid),
            ],
            "actions": ["python3 -m {} {} {}".format(module, num_solves, subdomain)],
            "targets": [beam.fom_test_set(config)],
            "clean": True,
        }


def task_proj_error():
    """Getting started: Projection error for FOM test sets"""
    module = "src.getting_started.projerr"
    code = SRC / "projerr.py"
    for distr in DISTR:
        for config in CONFIGS:
            yield {
                    "basename": f"proj_err_{beam.name}_{distr}_{config}",
                    "file_dep": [defs, code, beam.unit_cell_grid, beam.loc_pod_modes(distr, config), beam.fom_test_set(config)],
                    "actions": ["python3 -m {} {} {}".format(module, distr, config)],
                    "targets": [beam.proj_error(distr, config)],
                    "clean": True,
                    }


def task_plot_proj_error():
    """Getting started: Figure Projection Error"""
    module = "src.getting_started.plot_projerr"
    code = SRC / "plot_projerr.py"
    for config in beam.configurations:
        deps= [code]
        deps += [beam.proj_error(distr, config) for distr in beam.distributions]
        yield {
                "basename": f"fig_proj_err_{beam.name}_{config}",
                "file_dep": deps,
                "actions": ["python3 -m {}".format(module)],
                "targets": [beam.fig_proj_error(config)],
                "clean": True,
                }


def task_decomposition():
    """Getting started: Decompose local POD basis"""
    module = "src.getting_started.decompose_pod_basis"
    code = SRC / "decompose_pod_basis.py"
    for distr in DISTR:
        for config in CONFIGS:
            yield {
                "basename": f"decompose_{beam.name}_{distr}_{config}",
                "file_dep": [defs, code, beam.unit_cell_grid, beam.loc_pod_modes(distr, config)],
                "actions": ["python3 -m {} {} {}".format(module, distr, config)],
                "targets": [
                    beam.local_basis_npz(distr, config),
                    *with_h5(beam.fine_scale_modes_xdmf(distr, config)),
                    *with_h5(beam.pod_modes_xdmf(distr, config)),
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
