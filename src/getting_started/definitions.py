from typing import Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from multi.boundary import point_at, plane_at
from multi.problems import MultiscaleProblemDefinition
from dolfinx import default_scalar_type

ROOT = Path(__file__).parents[2]
WORK = ROOT / "work"


@dataclass
class BeamData:
    """Holds example specific parameters and manages filepaths.

    Args:
        name: The name of the example.
        gdim: The geometric dimension of the problem.
        length: The length of the beam.
        height: The height of the beam.
        nx: Number of coarse grid cells (subdomains) in x.
        ny: Number of coarse grid cells (subdomains) in y.
        resolution: `resolution ** 2` cells in each subdomain.
        geom_deg: Degree for geometry interpolation.
        fe_deg: FE degree.
        poisson_ratio: The poisson ratio of the material.
        youngs_modulus: The Young's modulus (reference value) of the material.
        mu_range: The value range of each parameter component.
        rrf_ttol: Target tolerance for range finder algo.
        rrf_ftol: Failure tolerance for range finder algo.
        rrf_num_testvecs: Number of test vectors for range finder algo.
        pod_rtol: Relative tolerance for POD algo.
        configurations: The configurations, i.e. oversampling problems.
        distributions: The distributions used in the randomized range finder.
        range_product: The inner product to use (rrf, projection error).
        lhs: Parameters for Latin-Hypercube-Sampling for each configuration.

    """

    name: str = "example"
    gdim: int = 2
    length: float = 10.0
    height: float = 1.0
    nx: int = 10
    ny: int = 1
    resolution: int = 10
    geom_deg: int = 1
    fe_deg: int = 2
    poisson_ratio: float = 0.3
    youngs_modulus: float = 20e3
    mu_range: tuple[float, float] = (1.0, 2.0)
    rrf_ttol: float = 5e-2
    rrf_ftol: float = 1e-15
    rrf_num_testvecs: int = 20
    pod_rtol: float = 1e-5
    configurations: tuple[str, str, str] = ("inner", "left", "right")
    distributions: tuple[str, str] = ("normal", "multivariate_normal")
    range_product: str = "h1"
    lhs: dict = field(
        default_factory=lambda: {
            "inner": {
                "name": "E",
                "ndim": 3,
                "samples": 30,
                "criterion": "center",
                "random_state": 1510,
            },
            "left": {
                "name": "E",
                "ndim": 2,
                "samples": 20,
                "criterion": "center",
                "random_state": 1511,
            },
            "right": {
                "name": "E",
                "ndim": 2,
                "samples": 20,
                "criterion": "center",
                "random_state": 1512,
            },
        }
    )

    def __post_init__(self):
        """create dirs"""
        self.grids_path.mkdir(exist_ok=True, parents=True)
        self.logs_path.mkdir(exist_ok=True, parents=True)
        for distr in self.distributions:
            self.bases_path(distr).mkdir(exist_ok=True, parents=True)

    @property
    def rf(self) -> Path:
        """run folder"""
        return WORK / f"{self.name}"

    @property
    def grids_path(self) -> Path:
        return self.rf / "grids"

    @property
    def logs_path(self) -> Path:
        return self.rf / "logs"

    def bases_path(self, distr: str) -> Path:
        return self.rf / f"bases/{distr}"

    @property
    def coarse_grid(self) -> Path:
        """Global coarse grid"""
        return self.grids_path / "coarse_grid.msh"

    @property
    def fine_grid(self) -> Path:
        """Global fine grid"""
        return self.grids_path / "fine_grid.xdmf"

    @property
    def unit_cell_grid(self) -> Path:
        return self.grids_path / "unit_cell.msh"

    def fine_oversampling_grid(self, configuration: str) -> Path:
        assert configuration in ("inner", "left", "right")
        return self.grids_path / f"fine_oversampling_grid_{configuration}.xdmf"

    @property
    def fom_displacement(self) -> Path:
        return self.rf / "fom_displacement.bp"

    @property
    def reduced_model(self) -> Path:
        """the global POD-ROM"""
        return self.rf / "reduced_model.out"

    @property
    def singular_values(self) -> Path:
        """singular values for the global POD-ROM"""
        return self.rf / "singular_values.npy"

    def log_range_approximation(self, distr: str, conf: str) -> Path:
        return self.logs_path / f"range_approximation_{distr}_{conf}.log"

    def log_projerr(self, distr: str, conf: str) -> Path:
        return self.logs_path / f"projerr_{distr}_{conf}.log"

    def log_decompose_pod_basis(self, distr: str, conf: str) -> Path:
        return self.logs_path / f"decompose_pod_basis_{distr}_{conf}.log"

    def log_extension(self, distr: str, cell: int) -> Path:
        return self.logs_path / f"extension_{distr}_{cell}.log"

    def log_run_locrom(self, distr: str) -> Path:
        return self.logs_path / f"run_locrom_{distr}.log"

    def loc_singular_values(self, distr: str, conf: str) -> Path:
        """singular values of POD compression for range approximation of parametric T"""
        return self.rf / f"loc_singular_values_{distr}_{conf}.npy"

    def loc_pod_modes(self, distr: str, conf: str) -> Path:
        """POD modes for range approximation of parametric T"""
        return self.rf / f"loc_pod_modes_{distr}_{conf}.npy"

    def pod_modes_bp(self, distr: str, conf: str) -> Path:
        """same as `loc_pod_modes` but adios2 (bp) format"""
        return self.rf / f"pod_modes_{distr}_{conf}.bp"

    def fine_scale_edge_modes_npz(self, distr: str, conf: str) -> Path:
        """edge-restricted fine scale part of pod modes"""
        return self.rf / f"fine_scale_edge_modes_{distr}_{conf}.npz"

    def fine_scale_modes_bp(self, distr: str, cell: int) -> Path:
        """fine scale basis functions after extension"""
        return self.rf / f"fine_scale_modes_{distr}_{cell}.bp"

    def fom_test_set(self, conf: str) -> Path:
        """test set generated from FOM solutions"""
        return self.rf / f"test_set_{conf}.npy"

    def proj_error(self, distr: str, conf: str) -> Path:
        """projection error for fom test set wrt pod basis"""
        return self.rf / f"proj_error_{distr}_{conf}.npy"

    def fig_proj_error(self, conf: str) -> Path:
        """figure of projection error plot"""
        return self.rf / f"fig_proj_error_{conf}.pdf"

    def fig_loc_svals(self, config: str) -> Path:
        """figure of singular values of POD compression after rrf"""
        return self.rf / f"fig_loc_svals_{config}.pdf"

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

    def local_basis_npz(self, distr: str, cell: int) -> Path:
        """final basis for loc rom assembly"""
        dir = self.bases_path(distr)
        return dir / f"basis_{cell:03}.npz"

    def loc_rom_error(self, distr: str) -> Path:
        """loc ROM error relative to FOM"""
        return self.rf / f"loc_rom_error_{distr}.csv"


class BeamProblem(MultiscaleProblemDefinition):
    def __init__(self, coarse_grid: str, fine_grid: str):
        super().__init__(coarse_grid, fine_grid)
        self.setup_coarse_grid(2)
        self.setup_fine_grid()
        self.build_edge_basis_config(self.cell_sets)

    def config_to_cell(self, config: str) -> int:
        """Maps config to global cell index."""
        map = {"inner": 4, "left": 0, "right": 9}
        return map[config]

    @property
    def cell_sets(self):
        """Returns cell sets for definition of edge basis configuration"""
        # the order is important
        # this way e.g. cell 1 will load modes for the left edge
        # from basis generated for cell 1 (config inner)
        cell_sets = {
            "inner": set([1, 2, 3, 4, 5, 6, 7, 8]),
            "left": set([0]),
            "right": set([9]),
        }
        return cell_sets

    @property
    def cell_sets_oversampling(self):
        """Returns cell sets that define oversampling domains"""
        # see preprocessing.py
        cells = {
            "inner": set([3, 4, 5]),
            "left": set([0, 1]),
            "right": set([8, 9]),
        }
        return cells

    @property
    def boundaries(self):
        x = self.coarse_grid.grid.geometry.x
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)
        return {
            "origin": (int(101), point_at([xmin[0], xmin[1], xmin[2]])),
            "bottom_right": (int(102), point_at([xmax[0], xmin[1], xmin[2]])),
        }

    def get_xmin_omega_in(self, cell_index: Optional[int] = None) -> np.ndarray:
        """Returns coordinate xmin of target subdomain"""
        if cell_index is not None:
            assert cell_index in (0, 4, 9)
        if cell_index == 0:
            xmin = np.array([[0.0, 0.0, 0.0]])
        elif cell_index == 4:
            xmin = np.array([[4.0, 0.0, 0.0]])
        elif cell_index == 9:
            xmin = np.array([[9.0, 0.0, 0.0]])
        else:
            raise NotImplementedError
        return xmin

    def get_dirichlet(self, cell_index: Optional[int] = None) -> Union[dict, None]:
        _, origin = self.boundaries["origin"]
        _, bottom_right = self.boundaries["bottom_right"]
        if cell_index is not None:
            assert cell_index in (0, 4, 9)
        if cell_index == 0:
            u_origin = np.array([0.0, 0.0], dtype=default_scalar_type)
            dirichlet = {"value": u_origin, "boundary": origin, "method": "geometrical"}
        elif cell_index == 4:
            dirichlet = None
        elif cell_index == 9:
            u_bottom_right = default_scalar_type(0.0)
            dirichlet = {
                "value": u_bottom_right,
                "boundary": bottom_right,
                "sub": 1,
                "entity_dim": 0,
                "method": "geometrical",
            }
        else:
            raise NotImplementedError
        return dirichlet

    def get_neumann(self, cell_index: Optional[int] = None):
        # is the same for all oversampling problems
        # see range_approximation.py
        return None

    def get_kernel_set(self, cell_index: int) -> tuple[int, ...]:
        """return indices of rigid body modes to be used"""
        if cell_index is not None:
            assert cell_index in (0, 4, 9)
        if cell_index == 0:
            # left, only rotation is free
            return (2,)
        elif cell_index == 4:
            # inner, use all rigid body modes
            return (0, 1, 2)
        elif cell_index == 9:
            # right, only trans y is constrained
            return (0, 2)

    def get_gamma_out(self, cell_index: Optional[int] = None) -> Callable:
        if cell_index is not None:
            assert cell_index in (0, 4, 9)
        if cell_index == 0:
            gamma_out = plane_at(2.0, "x")
        elif cell_index == 4:
            left = plane_at(3.0, "x")
            right = plane_at(6.0, "x")
            gamma_out = lambda x: np.logical_or(left(x), right(x))
        elif cell_index == 9:
            gamma_out = plane_at(8.0, "x")
        else:
            raise NotImplementedError
        return gamma_out


if __name__ == "__main__":
    data = BeamData(name="beam")
    problem = BeamProblem(data.coarse_grid.as_posix(), data.fine_grid.as_posix())
