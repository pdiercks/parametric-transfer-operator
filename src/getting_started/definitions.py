from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORK = ROOT / "work"


@dataclass
class Example:
    """Holds example specific data and manages filepaths.

    Args:
        name: The name of the example.
        nx: Number of cells in x.
        ny: Number of cells in y.
        resolution: `resolution ** 2` cells in each subdomain.
        fe_deg: FE degree.
        poisson_ratio: The poisson ratio of the material.
        youngs_modulus: The Young's modulus (reference value) of the material.
        mu_range: The value range of each parameter component.
        rrf_ttol: Target tolerance for range finder algo.
        rrf_ftol: Failure tolerance for range finder algo.
        rrf_num_testvecs: Number of test vectors for range finder algo.
        pod_rtol: Relative tolerance for POD algo.

    """
    name: str = "example"
    nx: int = 10
    ny: int = 1
    resolution: int = 10
    fe_deg: int = 2
    poisson_ratio: float = 0.3
    youngs_modulus: float = 20e3
    mu_range: tuple[float, float] = (1., 2.)
    rrf_ttol: float = 5e-2
    rrf_ftol: float = 1e-15
    rrf_num_testvecs: int = 20
    pod_rtol: float = 1e-6

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

    @property
    def coarse_oversampling_grid(self) -> Path:
        return self.grids_path / "coarse_oversampling_grid.msh"

    @property
    def fine_oversampling_grid(self) -> Path:
        return self.grids_path / "fine_oversampling_grid.xdmf"

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

    def range_approximation_log(self, distr: str) -> Path:
        return self.logs_path / f"range_approximation_{distr}.log"

    def loc_singular_values(self, distr: str) -> Path:
        """singular values of POD compression for range approximation of parametric T"""
        return self.rf / f"loc_singular_values_{distr}.npy"

    def loc_pod_modes(self, distr: str) -> Path:
        """POD modes for range approximation of parametric T"""
        return self.rf / f"loc_pod_modes_{distr}.npy"

    def pod_modes_xdmf(self, distr: str) -> Path:
        """same as `loc_pod_modes` but .xdmf format"""
        return self.rf / f"pod_modes_{distr}.xdmf"

    def fine_scale_modes_xdmf(self, distr: str) -> Path:
        """fine scale basis functions after extension"""
        return self.rf / f"fine_scale_modes_{distr}.xdmf"

    def local_basis_npz(self, distr: str) -> Path:
        """final local basis functions"""
        return self.rf / f"local_basis_{distr}.npz"

    def fom_test_set(self, subdomain_id: int) -> Path:
        """test set generated from FOM solutions"""
        return self.rf / f"test_set_{subdomain_id}.npy"
