from typing import Optional, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from multi.boundary import point_at, plane_at, within_range
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
                "samples": 100,
                "criterion": "center",
                "random_state": 1510,
            },
            "left": {
                "name": "E",
                "ndim": 2,
                "samples": 50,
                "criterion": "center",
                "random_state": 1510,
            },
            "right": {
                "name": "E",
                "ndim": 2,
                "samples": 50,
                "criterion": "center",
                "random_state": 1510,
            },
        }
    )

    def __post_init__(self):
        """create dirs"""
        self.grids_path.mkdir(exist_ok=True, parents=True)
        self.logs_path.mkdir(exist_ok=True, parents=True)

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
        return self.rf / "fom_displacement.xdmf"

    @property
    def reduced_model(self) -> Path:
        """the global POD-ROM"""
        return self.rf / "reduced_model.out"

    @property
    def singular_values(self) -> Path:
        """singular values for the global POD-ROM"""
        return self.rf / "singular_values.npy"

    def range_approximation_log(self, distr: str, conf: str) -> Path:
        return self.logs_path / f"range_approximation_{distr}_{conf}.log"

    def loc_singular_values(self, distr: str, conf: str) -> Path:
        """singular values of POD compression for range approximation of parametric T"""
        return self.rf / f"loc_singular_values_{distr}_{conf}.npy"

    def loc_pod_modes(self, distr: str, conf: str) -> Path:
        """POD modes for range approximation of parametric T"""
        return self.rf / f"loc_pod_modes_{distr}_{conf}.npy"

    def pod_modes_xdmf(self, distr: str, conf: str) -> Path:
        """same as `loc_pod_modes` but .xdmf format"""
        return self.rf / f"pod_modes_{distr}_{conf}.xdmf"

    def fine_scale_modes_xdmf(self, distr: str, conf: str) -> Path:
        """fine scale basis functions after extension"""
        return self.rf / f"fine_scale_modes_{distr}_{conf}.xdmf"

    def local_basis_npz(self, distr: str, conf: str) -> Path:
        """final local basis functions"""
        return self.rf / f"local_basis_{distr}_{conf}.npz"

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


class BeamProblem(MultiscaleProblemDefinition):
    def __init__(self, coarse_grid: str, fine_grid: str):
        super().__init__(coarse_grid, fine_grid)
        self.setup_coarse_grid(2)
        self.setup_fine_grid()

    def config_to_cell(self, config: str) -> int:
        """Maps config to global cell index."""
        map = {"inner": 4, "left": 0, "right": 9}
        return map[config]

    @property
    def cell_sets(self):
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

    def get_omega_in(self, cell_index: Optional[int] = None) -> Callable:
        if cell_index is not None:
            assert cell_index in (0, 4, 9)
        if cell_index == 0:
            marker = within_range([0.0, 0.0], [1.0, 1.0])
        elif cell_index == 4:
            marker = within_range([4.0, 0.0], [5.0, 1.0])
        elif cell_index == 9:
            marker = within_range([9.0, 0.0], [10.0, 1.0])
        else:
            raise NotImplementedError
        return marker

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
            # u_bottom_right = np.array([0], dtype=default_scalar_type) # raises RuntimeError: Rank mis-match between Constant and function space
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
    problem = BeamProblem(data.coarse_grid, data.fine_grid)
