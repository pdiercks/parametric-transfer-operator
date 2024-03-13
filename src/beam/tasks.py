"""tasks for the beam example"""

from .definitions import BeamData, ROOT
from pathlib import Path
import os
import shutil
from doit.tools import run_once

os.environ["PYMOR_COLORS_DISABLE"] = "1"
beam = BeamData(name="beam")
SRC = ROOT / "src" / f"{beam.name}"
CONFIGS = beam.configurations
DISTR = beam.distributions
NAMES = beam.training_strategies


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


def task_preprocessing():
    """Beam example: Preprocessing"""
    from .preprocessing import generate_meshes

    mesh_files = [beam.coarse_grid, beam.unit_cell_grid, *with_h5(beam.fine_grid)]
    for config in ("inner", "left", "right"):
        mesh_files += with_h5(beam.fine_oversampling_grid(config))

    return {
        "basename": f"preproc_{beam.name}",
        "file_dep": [SRC / "preprocessing.py"],
        "actions": [(generate_meshes, [beam])],
        "targets": mesh_files,
        "clean": True,
        "uptodate": [run_once],
    }


def task_mono_rom():
    """Beam example: Build monolithic ROM"""
    return {
        "basename": f"monolithic_rom_{beam.name}",
        "file_dep": [
            beam.coarse_grid,
            beam.unit_cell_grid,
            beam.fine_grid,
            SRC / "fom.py",
            SRC / "rom.py",
        ],
        "actions": [f"python3 -m src.{beam.name}.rom"],
        "targets": [beam.reduced_model, beam.singular_values],
        "clean": True,
    }


def task_hapod():
    """Beam example: Construct edge basis via HAPOD"""
    module = f"src.{beam.name}.range_approx_edge"
    file = SRC / "range_approx_edge.py"
    nworkers = 4  # number of workers in pool
    for distr in DISTR:
        for config in CONFIGS:
            yield {
                "name": f"{beam.name}_{distr}_{config}",
                "file_dep": [
                    file,
                    beam.fine_oversampling_grid(config),
                ],
                "actions": [
                    "python3 -m {} {} {} --max_workers {}".format(
                        module, distr, config, nworkers
                    )
                ],
                "targets": [
                    beam.log_edge_range_approximation(distr, config, "hapod"),
                    beam.fine_scale_edge_modes_npz(distr, config, "hapod"),
                    beam.loc_singular_values_npz(distr, config),
                    beam.hapod_rrf_bases_length(distr, config),
                ],
                "clean": [rm_rf],
            }


def task_heuristic_rrf():
    """Beam example: Construct edge basis via heuristic range finder"""
    module = f"src.{beam.name}.heuristic_range_approx"
    file = SRC / "heuristic_range_approx.py"
    for distr in DISTR:
        for config in CONFIGS:
            yield {
                "name": f"{beam.name}_{distr}_{config}",
                "file_dep": [
                    file,
                    beam.fine_oversampling_grid(config),
                ],
                "actions": [
                    "python3 -m {} {} {}".format(
                        module, distr, config
                    )
                ],
                "targets": [
                    beam.log_edge_range_approximation(distr, config, "heuristic"),
                    beam.fine_scale_edge_modes_npz(distr, config, "heuristic"),
                ],
                "clean": [rm_rf],
            }


def task_plot_loc_svals():
    """Beam example: Figure Singular Values"""
    module = f"src.{beam.name}.plot_svals"
    code = SRC / "plot_svals.py"
    for config in beam.configurations:
        deps = [code]
        deps += [
            beam.loc_singular_values_npz(distr, config) for distr in beam.distributions
        ]
        yield {
            "name": f"{beam.name}_{config}",
            "file_dep": deps,
            "actions": ["python3 -m {} %(targets)s {}".format(module, config)],
            "targets": [beam.fig_loc_svals(config)],
            "clean": True,
        }


def task_test_sets():
    """Beam example: Generate FOM test sets"""
    module = f"src.{beam.name}.fom_test_set"
    code = SRC / "fom_test_set.py"
    num_solves = 20
    map = {"inner": 4, "left": 0, "right": 9}
    for config in CONFIGS:
        subdomain = map[config]
        yield {
            "name": f"{beam.name}_{config}",
            "file_dep": [
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
    """Beam example: Projection error for FOM test sets"""
    module = f"src.{beam.name}.projerr"
    code = SRC / "projerr.py"

    for distr in DISTR:
        for config in CONFIGS:
            for name in NAMES:
                basis = beam.fine_scale_edge_modes_npz(distr, config, name)
                deps = [code, beam.unit_cell_grid, beam.fom_test_set(config)]
                deps.append(basis)
                yield {
                    "name": f"{name}_{distr}_{config}",
                    "file_dep": deps,
                    # TODO add basis as cli argument
                    "actions": ["python3 -m {} {} {} {}".format(module, distr, config, name)],
                    "targets": [
                        beam.proj_error(distr, config, name),
                        beam.log_projerr(distr, config, name),
                    ],
                    "clean": True,
                }


def task_plot_proj_error():
    """Beam example: Figure Projection Error"""
    module = f"src.{beam.name}.plot_projerr"
    code = SRC / "plot_projerr.py"
    for config in beam.configurations:
        for name in ["hapod", "heuristic"]:
            deps = [code]
            deps += [beam.proj_error(distr, config, name) for distr in beam.distributions]
            yield {
                "name": f"{name}_{config}",
                "file_dep": deps,
                "actions": ["python3 -m {} {} {} --output %(targets)s".format(module, config, name)],
                "targets": [beam.fig_proj_error(config, name)],
                "clean": True,
            }


def task_extension():
    """Beam example: Extend fine scale modes and write final basis"""
    module = f"src.{beam.name}.extension"
    num_cells = beam.nx * beam.ny
    for distr in DISTR:
        for name in NAMES:
            for cell_index in range(num_cells):
                config = beam.cell_to_config(cell_index)
                yield {
                        "name": f"{name}_{distr}_{cell_index}",
                        "file_dep": [beam.fine_scale_edge_modes_npz(distr, config, name)],
                        "actions": ["python3 -m {} {} {} {}".format(module, distr, name, cell_index)],
                        "targets": [beam.local_basis_npz(distr, name, cell_index), beam.fine_scale_modes_bp(distr, name, cell_index), beam.log_extension(distr, name, cell_index)],
                        "clean": [rm_rf],
                        }


def task_loc_rom():
    """Beam example: Validate the localized ROM."""
    module = f"src.{beam.name}.run_locrom"
    deps = [beam.fine_grid, beam.unit_cell_grid]
    num_cells = beam.nx * beam.ny
    num_test = 5 # TODO
    for distr in DISTR:
        for name in NAMES:
            deps += [beam.local_basis_npz(distr, name, cell) for cell in range(num_cells)]
            target = beam.loc_rom_error(distr, name)
            yield {
                    "name": f"{name}_{distr}",
                    "file_dep": deps,
                    "actions": ["python3 -m {} {} {} {} --output {}".format(module, distr, name, num_test, target)],
                    "targets": [target, beam.log_run_locrom(distr, name)],
                    "clean": True,
                    }


def task_plot_loc_rom_error():
    """Beam example: Plot localized ROM error."""
    module = f"src.{beam.name}.plot_locrom_error"
    return {
            "file_dep": [beam.loc_rom_error(d, name) for d in DISTR for name in NAMES],
            "actions": ["python3 -m {} %(targets)s".format(module)],
            "targets": [beam.fig_loc_rom_error],
            "clean": True,
            }


def task_optimize():
    """Beam example: Optimization"""
    optpy = ROOT / f"src/{beam.name}/optimization.py"
    return {
        "basename": f"optimize_{beam.name}",
        "file_dep": [optpy, beam.reduced_model],
        "actions": [f"python3 -m src.{beam.name}.optimization"],
        "verbosity": 2,
        "uptodate": [True],
    }


def task_plot_unit_cell_domain():
    """Beam example: plot unit cell domain"""

    def make_plot(targets):
        from mpi4py import MPI
        from dolfinx.io import gmshio
        from multi.postprocessing import plot_domain
        domain, _, _ = gmshio.read_from_msh(beam.unit_cell_grid.as_posix(), MPI.COMM_SELF, gdim=2)
        plot_domain(domain, cell_tags=None, transparent=False, colormap="bam-RdBu", output=targets[0])

    return {
            "file_dep": [beam.unit_cell_grid],
            "actions": [make_plot, "convert -trim %(targets)s %(targets)s"],
            "targets": [beam.fig_unit_cell],
            "uptodate": [True],
            "clean": True,
            }


def task_plot_global_domain():
    """Beam example: Plot global domain"""

    def make_plot(dependencies, targets):
        from mpi4py import MPI
        from dolfinx.io.utils import XDMFFile
        from multi.postprocessing import plot_domain
        xdmf_file = Path(dependencies[0]).with_suffix(".xdmf")
        with XDMFFile(MPI.COMM_WORLD, xdmf_file.as_posix(), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            cell_tags = xdmf.read_meshtags(domain, "subdomains")
        plot_domain(domain, cell_tags=cell_tags, transparent=False, colormap="bam-RdBu", output=targets[0])

    return {
            "file_dep": with_h5(beam.fine_grid),
            "actions": [make_plot, "convert -trim %(targets)s %(targets)s"],
            "targets": [beam.fig_fine_grid],
            "uptodate": [True],
            "clean": True,
            }


def task_fig_beam_sketch():
    """Beam example: Compile beam sketch"""
    target = ROOT / "figures/beam/beam_sketch.pdf"
    return {
            "file_dep": [SRC / "figures/beam_sketch.tex"],
            "actions": [f"latexmk -cd -pdf -outdir={target.parent} %(dependencies)s"],
            "targets": [target],
            "clean": True,
            }


def task_paper():
    """Beam example: Compile Paper"""
    source = ROOT / "paper/paper.tex"
    deps = [source]
    deps.append(beam.fig_loc_rom_error)
    deps.append(beam.fig_fine_grid)
    deps.append(beam.fig_unit_cell)
    deps.append(ROOT / "figures/beam/beam_sketch.pdf")
    for config in CONFIGS:
        deps.append(beam.fig_loc_svals(config))
        for name in NAMES:
            deps.append(beam.fig_proj_error(config, name))
    return {
        "file_dep": deps,
        "actions": ["latexmk -cd -pdf %s" % source],
        "targets": [source.with_suffix(".pdf")],
        "clean": ["latexmk -cd -C %s" % source],
    }


def task_show_paper():
    """Beam example: View Paper"""
    return {
        "file_dep": [ROOT / "paper/paper.pdf"],
        "actions": ["zathura %(dependencies)s"],
        "uptodate": [False],
    }


