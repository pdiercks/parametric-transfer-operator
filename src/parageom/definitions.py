"""ParaGeom example definitions."""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Callable, Optional

import dolfinx as df
from multi.boundary import within_range
import numpy as np
from pymor.parameters.base import Parameters

ROOT = Path(__file__).parents[2]
WORK = ROOT / "work"
SRC = Path(__file__).parent


@dataclass
class BeamData:
    """Holds example specific parameters and manages filepaths.

    Args:
        name: The name of the example.
        gdim: The geometric dimension of the problem.
        l_char: Characteristic length chosen to be the unit cell (subdomain) length.
        unit_length: Dimensionless unit length used to define computational domain(s).
        nx: Number of coarse grid cells (subdomains) in x.
        ny: Number of coarse grid cells (subdomains) in y.
        geom_deg: Degree for geometry interpolation.
        fe_deg: FE degree.
        poisson_ratio: The poisson ratio of the material.
        youngs_modulus: The Young's modulus (reference value) of the material.
        mu_range: The value range of each parameter component.
        mu_bar: Reference parameter value (Radius of parent domain).
        training_set_seed: Seed to generate seeds for training sets for each configuration (HAPOD, HRRF).
        testing_set_seed: Seed to generate seeds for testing sets for each configuration (HRRF).
        validation_set_seed: Seeding for the validation set (run_locrom.py).
        projerr_seed: Seeding for the testing set to compute the projection error.
        parameters: Dict of dict mapping parameter name to parameter dimension for each configuration etc.
        configurations: The configurations, i.e. oversampling problems.
        distributions: The distributions used in the randomized range finder.
        methods: Methods used for basis construction.
        range_product: The inner product to use (rrf, projection error).
        rrf_ttol: Target tolerance range finder.
        rrf_ftol: Failure tolerance range finder.
        rrf_num_testvecs: Number of testvectors range finder.
        run_mode: DEBUG or PRODUCTION mode. Affects mesh sizes, training set, realizations.

    """

    name: str = "parageom"
    gdim: int = 2
    l_char: float = 100.0  # [mm], characteristic length = unit length
    unit_length: float = 1.0  # dimensionless unit length
    nx: int = 10
    ny: int = 1
    geom_deg: int = 2
    fe_deg: int = 2
    poisson_ratio: float = 0.2
    youngs_modulus: float = 20e3  # [MPa]
    traction_y: float = 5.0  # [MPa]
    parameters: dict = field(
        default_factory=lambda: {
            "subdomain": Parameters({"R": 1}),
            "global": Parameters({"R": 10}),
            "left": Parameters({"R": 3}),
            "right": Parameters({"R": 3}),
            "inner": Parameters({"R": 4}),
        }
    )
    training_set_seed: int = 767667058
    testing_set_seed: int = 545445836
    validation_set_seed: int = 986718877
    projerr_seed: int = 923719053
    configurations: tuple[str, str, str] = ("left", "inner", "right")
    distributions: tuple[str, ...] = ("normal",)
    methods: tuple[str, ...] = ("hapod", "heuristic")
    epsilon_star: dict = field(
            default_factory=lambda: {
                "heuristic": 0.01,
                "hapod": 0.01,
                })
    epsilon_star_projerr: float = 0.001
    omega: float = 0.5
    rrf_ttol: float = 10e-2
    rrf_ftol: float = 1e-10
    rrf_num_testvecs: int = 20
    neumann_rtol: float = 1e-5
    run_mode: str = "DEBUG"

    def __post_init__(self):
        """Creates directory structure and dependent attributes"""

        self.length: float = self.unit_length * self.nx
        self.height: float = self.unit_length * self.ny
        a = self.unit_length
        self.mu_range: tuple[float, float] = (0.1 * a, 0.3 * a)  # [mm]
        self.mu_bar: float = 0.2 * a  # [mm]

        self.grids_path.mkdir(exist_ok=True, parents=True)
        self.figures_path.mkdir(exist_ok=True, parents=True)

        # Have a separate folder for each configuration to store
        # the physical oversampling meshes.
        # naming convention: oversampling_{index}.msh
        # store the training set, such that via {index} the
        # parameter value can be determined
        for config in list(self.configurations) + ["global", "target"]:
            p = self.grids_path / config
            p.mkdir(exist_ok=True, parents=True)

        # NOTE realizations
        # We need to compute several realizations since we are using a randomized method.
        # To be able to compare the different ROMs (sets of basis functions), the training
        # set should be the same for all realizations. Therefore, the same physical meshes
        # are used for each realization.

        if self.run_mode == "DEBUG":
            self.num_real = 1
            self.num_intervals = 12
        elif self.run_mode == "PRODUCTION":
            self.num_real = 20
            self.num_intervals = 20
        else:
            raise NotImplementedError

        for nr in range(self.num_real):
            self.real_folder(nr).mkdir(exist_ok=True, parents=True)
            for name in self.methods:
                # self.method_folder(nr, name).mkdir(exist_ok=True, parents=True)
                self.logs_path(nr, name).mkdir(exist_ok=True, parents=True)
                self.bases_path(nr, name).mkdir(exist_ok=True, parents=True)
            (self.method_folder(nr, "hapod") / "pod_modes").mkdir(
                exist_ok=True, parents=True
            )
            (self.method_folder(nr, "heuristic") / "modes").mkdir(
                exist_ok=True, parents=True
            )

    @property
    def plotting_style(self) -> Path:
        """eccomas proceedings mplstyle"""
        return ROOT / "src/proceedings.mplstyle"

    @property
    def figures_path(self) -> Path:
        return ROOT / "figures" / f"{self.name}"

    @property
    def rf(self) -> Path:
        """run folder"""
        return WORK / f"{self.name}"

    @property
    def grids_path(self) -> Path:
        return self.rf / "grids"

    def real_folder(self, nr: int) -> Path:
        """realization folder"""
        return self.rf / f"realization_{nr:02}"

    def method_folder(self, nr: int, name: str) -> Path:
        """training strategy / method folder"""
        return self.real_folder(nr) / name

    def logs_path(self, nr: int, name: str) -> Path:
        return self.method_folder(nr, name) / "logs"

    def bases_path(self, nr: int, name: str, distr: str = "normal") -> Path:
        """
        Args:
            nr: realization index.
            name: name of the training strategy / method.
            distr: distribution.
        """
        return self.method_folder(nr, name) / f"bases/{distr}"

    def training_set(self, config: str) -> Path:
        """Write training set as numpy array"""
        return self.grids_path / config / "training_set.out"

    def coarse_grid(self, config: str) -> Path:
        """Global coarse grid"""
        assert config in list(self.configurations) + ["global", "target"]
        return self.grids_path / config / "coarse_grid.msh"

    def parent_domain(self, config: str) -> Path:
        assert config in list(self.configurations) + ["global", "target"]
        return self.grids_path / config / "parent_domain.msh"

    @property
    def parent_unit_cell(self) -> Path:
        return self.grids_path / "parent_unit_cell.msh"

    @property
    def cell_type(self) -> str:
        """The cell type of the parent unit cell mesh."""
        match self.geom_deg:
            case 1:
                return "quad"
            case 2:
                return "quad9"
            case _:
                return "quad"

    def archetype_to_cell(self, atype: int) -> int:
        atc = {0: 0, 1: 1, 2: 4, 3: 8, 4: 9}
        return atc[atype]

    def config_to_cell(self, config: str) -> int:
        """Maps config to global cell index."""
        map = {"inner": 4, "left": 0, "right": 9}
        return map[config]

    def config_to_target_cell(self, config: str) -> int:
        """Maps config to cell index of target subdomain."""
        map = {"inner": 1, "left": 0, "right": 1}
        return map[config]

    def ntrain(self, config: str) -> int:
        """Define size of training set"""
        if self.run_mode == "DEBUG":
            map = {"left": 60, "inner": 80, "right": 60}
            return map[config]
        elif self.run_mode == "PRODUCTION":
            map = {"left": 40, "inner": 60, "right": 40}
            return map[config]
        else:
            raise NotImplementedError

    def cell_to_config(self, cell: int) -> str:
        """Maps global cell index to config."""
        assert cell in list(range(self.nx * self.ny))
        map = {0: "left", 9: "right"}
        config = map.get(cell, "inner")
        return config

    def local_basis_npy(self, nr: int, name: str, distr: str, cell: int) -> Path:
        """final basis for loc rom assembly"""
        dir = self.bases_path(nr, name, distr)
        return dir / f"basis_{cell:02}.npy"

    def local_basis_dofs_per_vert(self, nr: int, method: str, distr: str) -> Path:
        dir = self.bases_path(nr, method, distr)
        return dir / "dofs_per_vert.npy"

    def locrom_error(self, nreal: int, method: str, distr: str, ei: bool=False) -> Path:
        """loc ROM error"""
        dir = self.method_folder(nreal, method)
        if ei:
            return dir / f"locrom_error_ei_{distr}.npz"
        else:
            return dir / f"locrom_error_{distr}.npz"

    # @property
    # def fom_minimization_data(self) -> Path:
    #     """FOM minimization data"""
    #     return self.rf / "fom_minimization_data.out"
    #
    # def rom_minimization_data(self, distr: str, name: str) -> Path:
    #     """ROM minimization data"""
    #     return self.rf / f"rom_minimization_data_{distr}_{name}.out"
    #
    # @property
    # def minimization_data_table(self) -> Path:
    #     return self.rf / "minimization_data.csv"
    #
    # @property
    # def minimization_comparison_table(self) -> Path:
    #     return self.rf / "minimization_comparison.csv"
    #
    # @property
    # def fig_fom_opt(self) -> Path:
    #     return self.figures_path / "fig_fom_opt.pdf"
    #
    # @property
    # def fig_rom_opt(self) -> Path:
    #     return self.figures_path / "fig_rom_opt.pdf"
    def fig_projerr(self, config: str) -> Path:
        return self.figures_path / f"fig_projerr_{config}.pdf"

    @property
    def fig_locrom_error(self) -> Path:
        return self.figures_path / "locrom_error.pdf"

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

    def log_basis_construction(
        self, nr: int, method: str, distr: str, config: str
    ) -> Path:
        return self.logs_path(nr, method) / f"basis_construction_{distr}_{config}.log"

    def log_projerr(self, nr: int, method: str, distr: str, config: str) -> Path:
        return self.logs_path(nr, method) / f"projerr_{distr}_{config}.log"

    def log_gfem(self, nr: int, method: str, distr: str) -> Path:
        return self.logs_path(nr, method) / f"gfem_{distr}.log"

    def log_run_locrom(self, nr: int, method: str, distr: str, ei: bool=False) -> Path:
        dir = self.logs_path(nr, method)
        if ei:
            return dir / f"run_locrom_ei_{distr}.log"
        else:
            return dir / f"run_locrom_{distr}.log"

    def hapod_singular_values(self, nr: int, distr: str, conf: str) -> Path:
        """singular values of final POD"""
        return self.method_folder(nr, "hapod") / f"singular_values_{distr}_{conf}.npy"

    def hapod_modes_xdmf(self, nr: int, distr: str, config: str) -> Path:
        """modes of the final POD"""
        dir = self.method_folder(nr, "hapod") / "pod_modes"
        return dir / f"modes_{distr}_{config}.xdmf"

    def heuristic_modes_xdmf(self, nr: int, distr: str, config: str) -> Path:
        """modes computed by heuristic range finder"""
        dir = self.method_folder(nr, "heuristic") / "modes"
        return dir / f"modes_{distr}_{config}.xdmf"

    def hapod_modes_npy(self, nr: int, distr: str, config: str) -> Path:
        """modes of the final POD"""
        dir = self.method_folder(nr, "hapod") / "pod_modes"
        return dir / f"modes_{distr}_{config}.npy"

    def heuristic_modes_npy(self, nr: int, distr: str, config: str) -> Path:
        """modes computed by heuristic range finder"""
        dir = self.method_folder(nr, "heuristic") / "modes"
        return dir / f"modes_{distr}_{config}.npy"

    def projerr(self, nr: int, method: str, distr: str, config: str) -> Path:
        dir = self.method_folder(nr, method)
        return dir / f"projerr_{distr}_{config}.npz"

    @property
    def target_subdomain(self) -> Path:
        return self.parent_domain("target")

    def config_to_omega_in(self, config: str, local=True) -> list[int]:
        """Maps config to cell local index/indices of oversampling domain that correspond to omega in."""
        global_indices = {"left": [0, 1], "right": [8, 9], "inner": [4, 5]}
        # FIXME: this is not the case maybe?
        local_indices = {"left": [0, 1], "right": [1, 2], "inner": [1, 2]}
        if local:
            return local_indices[config]
        else:
            return global_indices[config]

    # FIXME: not needed for GFEM, but I may use edge functions as well.
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
    # FIXME: is this used anywhere?
    def cell_sets_oversampling(self):
        """Returns cell sets that define oversampling domains"""
        # see preprocessing.py
        cells = {
            "inner": set([3, 4, 5, 6]),
            "left": set([0, 1]),
            "right": set([8, 9]),
        }
        return cells

    def boundaries(self, domain: df.mesh.Mesh):
        """Returns Dirichlet boundaries of the global domain.

        Args:
            domain: The global domain.

        Note:
            This only defines markers and should be used with `df.mesh.locate_entities_boundary`.

        """

        x = domain.geometry.x
        a = self.unit_length
        xmin = np.amin(x, axis=0)
        xmax = np.amax(x, axis=0)

        return {
            "support_left": (
                int(101),
                within_range([xmin[0], xmin[1], xmin[2]], [a / 2, xmin[1], xmin[2]]),
            ),
            "support_right": (
                int(102),
                within_range(
                    [xmax[0] - a / 2, xmin[1], xmin[2]], [xmax[0], xmin[1], xmin[2]]
                ),
            ),
        }

    def get_dirichlet(
            self, domain: df.mesh.Mesh, config: str
    ) -> Optional[dict]:
        boundaries = self.boundaries(domain)
        _, left = boundaries["support_left"]
        _, right = boundaries["support_right"]

        # NOTE
        # this only defines markers using `within_range`
        # code needs to use df.mesh.locate_entities_boundary

        if config == "left":
            u_origin = (df.default_scalar_type(0.0), df.default_scalar_type(0.0))
            dirichlet = {
                "value": u_origin,
                "boundary": left,
                "entity_dim": 1,
                "sub": None,
            }
        elif config == "inner":
            dirichlet = None
        elif config == "right":
            u_bottom_right = df.default_scalar_type(0.0)
            dirichlet = {
                "value": u_bottom_right,
                "boundary": right,
                "entity_dim": 1,
                "sub": 1,
            }
        else:
            raise NotImplementedError
        return dirichlet

    @property
    def get_neumann(self) -> None:
        return None

    def get_kernel_set(self, cell_index: int) -> tuple[int, ...]:
        """return indices of rigid body modes to be used"""
        if cell_index in (0, 1):
            # left, only rotation is free
            return (2,)
        elif cell_index in (4, 5):
            # inner, use all rigid body modes
            return (0, 1, 2)
        elif cell_index in (8, 9):
            # right, only trans y is constrained
            return (0, 2)
        else:
            raise NotImplementedError

    def get_gamma_out(self, cell_index: Optional[int] = None) -> Callable:
        unit_length = self.unit_length
        y = self.height
        tol = 1e-4

        # NOTE
        # this only defines the marker
        # code needs to use df.mesh.locate_entities_boundary

        if cell_index in (0, 1):
            x = 3 * unit_length
            start = [x, 0.0 + tol, 0.0]
            end = [x, y - tol, 0.0]
            gamma_out = within_range(start, end)
        elif cell_index in (4, 5):
            x_left = 3 * unit_length
            x_right = 7 * unit_length
            left = within_range([x_left, 0.0 + tol, 0.0], [x_left, y - tol, 0.0])
            right = within_range([x_right, 0.0 + tol, 0.0], [x_right, y - tol, 0.0])

            def gamma_out(x):
                return np.logical_or(left(x), right(x))
        elif cell_index in (8, 9):
            x = 7 * unit_length
            gamma_out = within_range([x, 0.0 + tol, 0.0], [x, y - tol, 0.0])
        else:
            raise NotImplementedError
        return gamma_out
