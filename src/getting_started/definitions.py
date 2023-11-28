from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[2]
WORK = ROOT / "work"

@dataclass
class Example:
    """Holds example specific data.

    Args:
        name: The name of the example.
        nx: Number of cells in x.
        ny: Number of cells in y.
        resolution: `resolution ** 2` cells in each subdomain.
        fe_deg: FE degree.

    """
    name: str = "example"
    nx: int = 10
    ny: int = 1
    resolution: int = 4
    fe_deg: int = 1

    @property
    def rf(self) -> Path:
        """run folder"""
        return WORK / f"{self.name}"

    @property
    def grids_path(self) -> Path:
        return self.rf / "grids"

    @property
    def coarse_grid(self) -> Path:
        """Global coarse grid"""
        return self.grids_path / "global/coarse_grid.msh"

    @property
    def fine_grid(self) -> Path:
        """Global fine grid"""
        return self.grids_path / "global/fine_grid.xdmf"

    @property
    def unit_cell_grid(self) -> Path:
        return self.grids_path / "unit_cell.msh"
