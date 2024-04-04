# from typing import Optional, Callable, Union
from dataclasses import dataclass
# from dataclasses import field
from pathlib import Path
import numpy as np
# from multi.boundary import point_at, plane_at
# from multi.problems import MultiscaleProblemDefinition
# from dolfinx import default_scalar_type
from pymor.parameters.base import Parameters

ROOT = Path(__file__).parents[2]
WORK = ROOT / "work"
SRC = Path(__file__).parent


@dataclass
class BeamData:
    """Holds example specific parameters and manages filepaths.

    Args:
        name: The name of the example.
        num_real: The number of realizations of this example.
        gdim: The geometric dimension of the problem.
        length: The length of the beam.
        height: The height of the beam.
        nx: Number of coarse grid cells (subdomains) in x.
        ny: Number of coarse grid cells (subdomains) in y.
        num_intervals: Number of (interval) cells per edge.
        geom_deg: Degree for geometry interpolation.
        fe_deg: FE degree.
        poisson_ratio: The poisson ratio of the material.
        youngs_modulus: The Young's modulus (reference value) of the material.
        parameter_names: The names of all parameters.
        parameter_dims: The number of components of each parameter.
        mu_range: The value range of each parameter component.
        parameters: Dict mapping parameter name to parameter dimension.

    """

    name: str = "parageom"
    num_real: int = 1
    gdim: int = 2
    length: float = 10.0
    height: float = 1.0
    nx: int = 10
    ny: int = 1
    num_intervals: int = 12
    geom_deg: int = 1
    fe_deg: int = 2
    poisson_ratio: float = 0.3
    youngs_modulus: float = 20e3
    parameter_name: str = "R"
    parameter_dims: tuple[int, int] = (1, 10)
    mu_range: tuple[float, float] = (0.1, 0.3)

    def __post_init__(self):
        self.grids_path.mkdir(exist_ok=True, parents=True)
        self.logs_path.mkdir(exist_ok=True, parents=True)
        self.figures_path.mkdir(exist_ok=True, parents=True)
        pname = self.parameter_name
        dims = self.parameter_dims
        sub_param = eval('{ "%s" : %d }' % (pname, dims[0]))
        global_param = eval('{ "%s" : %d }' % (pname, dims[1]))
        self.parameters = {"subdomain": Parameters(sub_param), "global": Parameters(global_param)}

    @property
    def plotting_style(self) -> Path:
        """eccomas proceedings mplstyle"""
        return ROOT / "src/proceedings.mplstyle"

    @property
    def rf(self) -> Path:
        """run folder"""
        return WORK / f"{self.name}"

    @property
    def figures_path(self) -> Path:
        return ROOT / "figures" / f"{self.name}"

    @property
    def grids_path(self) -> Path:
        return self.rf / "grids"

    @property
    def logs_path(self) -> Path:
        return self.rf / "logs"

    def bases_path(self, distr: str, name: str) -> Path:
        return self.rf / f"bases/{distr}/{name}"

    @property
    def coarse_grid(self) -> Path:
        """Global coarse grid"""
        return self.grids_path / "coarse_grid.msh"

    @property
    def parent_unit_cell(self) -> Path:
        return self.grids_path / "parent_unit_cell.msh"

    def config_to_cell(self, config: str) -> int:
        """Maps config to global cell index."""
        map = {"inner": 4, "left": 0, "right": 9}
        return map[config]

    def cell_to_config(self, cell: int) -> str:
        """Maps global cell index to config."""
        assert cell in list(range(self.nx * self.ny))
        map = {0: "left", 9: "right"}
        config = map.get(cell, "inner")
        return config

    def local_basis_npz(self, distr: str, name: str, cell: int) -> Path:
        """final basis for loc rom assembly"""
        dir = self.bases_path(distr, name)
        return dir / f"basis_{cell:03}.npz"

    def loc_rom_error(self, distr: str, name: str) -> Path:
        """loc ROM error relative to FOM"""
        return self.rf / f"loc_rom_error_{distr}_{name}.csv"

    @property
    def fom_minimization_data(self) -> Path:
        """FOM minimization data"""
        return self.rf / "fom_minimization_data.out"

    def rom_minimization_data(self, distr: str, name: str) -> Path:
        """ROM minimization data"""
        return self.rf / f"rom_minimization_data_{distr}_{name}.out"

    @property
    def minimization_data_table(self) -> Path:
        return self.rf / "minimization_data.csv"

    @property
    def minimization_comparison_table(self) -> Path:
        return self.rf / "minimization_comparison.csv"

    @property
    def fig_fom_opt(self) -> Path:
        return self.figures_path / "fig_fom_opt.pdf"

    @property
    def fig_rom_opt(self) -> Path:
        return self.figures_path / "fig_rom_opt.pdf"

    @property
    def realizations(self) -> Path:
        """Returns realizations that can be used to create ``np.random.SeedSequence``"""
        file = SRC / "realizations.npy"
        if not file.exists():
            self._generate_realizations(file)
        return file

    def _generate_realizations(self, outpath: Path) -> None:
        seed = np.random.SeedSequence()
        realizations = seed.generate_state(self.num_real)
        np.save(outpath, realizations)


# class BeamProblem(MultiscaleProblemDefinition):
#     def __init__(self, coarse_grid: str, fine_grid: str):
#         super().__init__(coarse_grid, fine_grid)
#         self.setup_coarse_grid(2)
#         self.setup_fine_grid()
#         self.build_edge_basis_config(self.cell_sets)
#
#     def config_to_cell(self, config: str) -> int:
#         """Maps config to global cell index."""
#         map = {"inner": 4, "left": 0, "right": 9}
#         return map[config]
#
#     @property
#     def cell_sets(self):
#         """Returns cell sets for definition of edge basis configuration"""
#         # the order is important
#         # this way e.g. cell 1 will load modes for the left edge
#         # from basis generated for cell 1 (config inner)
#         cell_sets = {
#             "inner": set([1, 2, 3, 4, 5, 6, 7, 8]),
#             "left": set([0]),
#             "right": set([9]),
#         }
#         return cell_sets
#
#     @property
#     def cell_sets_oversampling(self):
#         """Returns cell sets that define oversampling domains"""
#         # see preprocessing.py
#         cells = {
#             "inner": set([3, 4, 5]),
#             "left": set([0, 1]),
#             "right": set([8, 9]),
#         }
#         return cells
#
#     @property
#     def boundaries(self):
#         x = self.coarse_grid.grid.geometry.x
#         xmin = np.amin(x, axis=0)
#         xmax = np.amax(x, axis=0)
#         return {
#             "origin": (int(101), point_at([xmin[0], xmin[1], xmin[2]])),
#             "bottom_right": (int(102), point_at([xmax[0], xmin[1], xmin[2]])),
#         }
#
#     def get_xmin_omega_in(self, cell_index: Optional[int] = None) -> np.ndarray:
#         """Returns coordinate xmin of target subdomain"""
#         if cell_index is not None:
#             assert cell_index in (0, 4, 9)
#         if cell_index == 0:
#             xmin = np.array([[0.0, 0.0, 0.0]])
#         elif cell_index == 4:
#             xmin = np.array([[4.0, 0.0, 0.0]])
#         elif cell_index == 9:
#             xmin = np.array([[9.0, 0.0, 0.0]])
#         else:
#             raise NotImplementedError
#         return xmin
#
#     def get_dirichlet(self, cell_index: Optional[int] = None) -> Union[dict, None]:
#         _, origin = self.boundaries["origin"]
#         _, bottom_right = self.boundaries["bottom_right"]
#         if cell_index is not None:
#             assert cell_index in (0, 4, 9)
#         if cell_index == 0:
#             u_origin = np.array([0.0, 0.0], dtype=default_scalar_type)
#             dirichlet = {"value": u_origin, "boundary": origin, "method": "geometrical"}
#         elif cell_index == 4:
#             dirichlet = None
#         elif cell_index == 9:
#             u_bottom_right = default_scalar_type(0.0)
#             dirichlet = {
#                 "value": u_bottom_right,
#                 "boundary": bottom_right,
#                 "sub": 1,
#                 "entity_dim": 0,
#                 "method": "geometrical",
#             }
#         else:
#             raise NotImplementedError
#         return dirichlet
#
#     def get_neumann(self, cell_index: Optional[int] = None):
#         # is the same for all oversampling problems
#         return None
#
#     def get_kernel_set(self, cell_index: int) -> tuple[int, ...]:
#         """return indices of rigid body modes to be used"""
#         if cell_index is not None:
#             assert cell_index in (0, 4, 9)
#         if cell_index == 0:
#             # left, only rotation is free
#             return (2,)
#         elif cell_index == 4:
#             # inner, use all rigid body modes
#             return (0, 1, 2)
#         elif cell_index == 9:
#             # right, only trans y is constrained
#             return (0, 2)
#
#     def get_gamma_out(self, cell_index: Optional[int] = None) -> Callable:
#         if cell_index is not None:
#             assert cell_index in (0, 4, 9)
#         if cell_index == 0:
#             gamma_out = plane_at(2.0, "x")
#         elif cell_index == 4:
#             left = plane_at(3.0, "x")
#             right = plane_at(6.0, "x")
#             gamma_out = lambda x: np.logical_or(left(x), right(x))
#         elif cell_index == 9:
#             gamma_out = plane_at(8.0, "x")
#         else:
#             raise NotImplementedError
#         return gamma_out


if __name__ == "__main__":
    data = BeamData(name="beam")
    realizations = np.load(data.realizations)
    print(realizations)
    # problem = BeamProblem(data.coarse_grid.as_posix(), data.fine_grid.as_posix())
