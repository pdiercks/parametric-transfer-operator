"""ParaGeom example definitions."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import dolfinx as df
import numpy as np
from multi.boundary import plane_at, point_at, within_range
from pymor.parameters.base import Parameters

ROOT = Path(__file__).parents[2]
WORK = ROOT / 'work'
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
        distributions: The distributions used in the randomized range finder.
        methods: Methods used for basis construction.
        range_product: The inner product to use (rrf, projection error).
        rrf_ttol: Target tolerance range finder.
        rrf_ftol: Failure tolerance range finder.
        rrf_num_testvecs: Number of testvectors range finder.
        debug: Run in debug mode.

    """

    name: str = 'parageom'
    gdim: int = 2
    characteristic_length = 100.0  # [mm]
    characteristic_displacement = 0.1  # [mm]
    unit_length: float = 1.0  # dimensionless unit length
    nx: int = 10
    ny: int = 1
    geom_deg: int = 2
    fe_deg: int = 2
    poisson_ratio: float = 0.2
    youngs_modulus: float = 30e3  # [MPa]
    plane_stress: bool = True
    traction_y: float = 0.0375  # [MPa]
    # TODO extent this for all k -> use list or array
    parameter_name: str = 'R'
    parameter_dim: tuple[int, ...] = (2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2)
    parameters: dict = field(
        default_factory=lambda: {
            'subdomain': Parameters({'R': 1}),
            'global': Parameters({'R': 10}),
            'left': Parameters({'R': 3}),
            'right': Parameters({'R': 3}),
            'inner': Parameters({'R': 4}),
        }
    )
    training_set_seed: int = 767667058
    testing_set_seed: int = 545445836
    validation_set_seed: int = 241690
    projerr_seed: int = 923719053
    distributions: tuple[str, ...] = ('normal',)
    methods: tuple[str, ...] = ('hapod', 'heuristic')
    epsilon_star: dict = field(
        default_factory=lambda: {
            'heuristic': 0.001,
            'hapod': 0.001,
        }
    )
    epsilon_star_projerr: float = 0.001
    omega: float = 0.5  # ω related to HAPOD (not output functional)
    rrf_ttol: float = 10e-2
    rrf_ftol: float = 1e-10
    rrf_num_testvecs: int = 20
    neumann_rtol: float = 1e-8  # hrrf pod extension
    neumann_tag: int = 194
    mdeim_rtol: float = 1e-5
    debug: bool = False
    validate_rom: dict = field(default_factory=lambda: {'num_params': 200, 'num_modes': list(range(20, 101, 20))})

    def __post_init__(self):
        """Creates directory structure and dependent attributes."""
        a = self.unit_length
        self.length: float = a * self.nx
        self.height: float = a * self.ny
        self.mu_range: tuple[float, float] = (0.1 * a, 0.3 * a)  # [mm]
        self.mu_bar: float = 0.2 * a  # [mm]
        E = self.youngs_modulus
        NU = self.poisson_ratio
        self.λ = E * NU / (1 + NU) / (1 - 2 * NU)
        self.μ = E / 2 / (1 + NU)
        # μ_c = 2.4μ, such that E/μ_c = 100
        self.characteristic_mu = 0.024 * self.μ
        self.E = E / self.characteristic_mu
        self.NU = NU
        self.sigma_scale = self.characteristic_length / (self.characteristic_mu * self.characteristic_displacement)

        self.grids_path.mkdir(exist_ok=True, parents=True)
        self.figures_path.mkdir(exist_ok=True, parents=True)

        for config in ['global']:
            p = self.grids_path / config
            p.mkdir(exist_ok=True, parents=True)

        # NOTE realizations
        # We need to compute several realizations since we are using a randomized method.
        # To be able to compare the different ROMs (sets of basis functions), the training
        # set should be the same for all realizations. Therefore, the same physical meshes
        # are used for each realization.

        if self.debug:
            self.num_real = 1
            self.num_intervals = 12
        else:
            self.num_real = 20
            self.num_intervals = 20

        for nr in range(self.num_real):
            self.real_folder(nr).mkdir(exist_ok=True, parents=True)
            for name in self.methods:
                # self.method_folder(nr, name).mkdir(exist_ok=True, parents=True)
                self.logs_path(nr, name).mkdir(exist_ok=True, parents=True)
                self.bases_path(nr, name).mkdir(exist_ok=True, parents=True)
            (self.method_folder(nr, 'hapod') / 'pod_modes').mkdir(exist_ok=True, parents=True)
            (self.method_folder(nr, 'heuristic') / 'modes').mkdir(exist_ok=True, parents=True)

    @property
    def plotting_style(self) -> Path:
        """Eccomas proceedings mplstyle."""
        return ROOT / 'src/proceedings.mplstyle'

    @property
    def figures_path(self) -> Path:
        return ROOT / 'figures' / f'{self.name}'

    @property
    def rf(self) -> Path:
        """Run folder."""
        return WORK / f'{self.name}'

    @property
    def grids_path(self) -> Path:
        return self.rf / 'grids'

    def real_folder(self, nr: int) -> Path:
        """Realization folder."""
        return self.rf / f'realization_{nr:02}'

    def method_folder(self, nr: int, name: str) -> Path:
        """Training strategy / method folder."""
        return self.real_folder(nr) / name

    def logs_path(self, nr: int, name: str) -> Path:
        return self.method_folder(nr, name) / 'logs'

    def bases_path(self, nr: int, name: str, distr: str = 'normal') -> Path:
        """Return Path to bases folder.

        Args:
            nr: realization index.
            name: name of the training strategy / method.
            distr: distribution.

        """
        return self.method_folder(nr, name) / f'bases/{distr}'

    def training_set(self, config: str) -> Path:
        """Write training set as numpy array."""
        return self.grids_path / config / 'training_set.out'

    def coarse_grid(self, config: str) -> Path:
        """Global coarse grid."""
        assert config in ['global']
        return self.grids_path / config / 'coarse_grid.msh'

    def parent_domain(self, config: str) -> Path:
        assert config in ['global']
        return self.grids_path / config / 'parent_domain.msh'

    @property
    def parent_unit_cell(self) -> Path:
        return self.grids_path / 'parent_unit_cell.msh'

    @property
    def singular_values_auxiliary_problem(self) -> Path:
        return self.rf / 'singular_values_auxiliary_problem.npy'

    @property
    def cell_type(self) -> str:
        """The cell type of the parent unit cell mesh."""
        match self.geom_deg:
            case 1:
                return 'quad'
            case 2:
                return 'quad9'
            case _:
                return 'quad'

    def archetype_to_cell(self, atype: int) -> int:
        atc = {0: 0, 1: 1, 2: 4, 3: 8, 4: 9}
        return atc[atype]

    def config_to_cell(self, config: str) -> int:
        """Maps config to global cell index."""
        map = {'inner': 4, 'left': 0, 'right': 9}
        return map[config]

    def config_to_target_cell(self, config: str) -> int:
        """Maps config to cell index of target subdomain."""
        map = {'inner': 1, 'left': 0, 'right': 1}
        return map[config]

    def ntrain(self, method: str, k: int) -> int:
        """Define size of training set for k-th transfer problem."""
        if method == 'hapod':
            if k in (0, 10):
                return 100
            elif k in (1, 9):
                return 150
            else:
                return 200
        elif method == 'heuristic':
            if k in (0, 10):
                return 30
            elif k in (1, 9):
                return 40
            else:
                return 50
        else:
            raise NotImplementedError

    def ntest(self, method: str, k: int) -> int:
        if method == 'heuristic':
            if k in (0, 10):
                return 100
            elif k in (1, 9):
                return 150
            else:
                return 200
        else:
            raise NotImplementedError

    def cell_to_config(self, cell: int) -> str:
        """Maps global cell index to config."""
        assert cell in list(range(self.nx * self.ny))
        map = {0: 'left', 9: 'right'}
        config = map.get(cell, 'inner')
        return config

    def local_basis_npy(self, nr: int, cell: int, method='hapod', distr='normal') -> Path:
        """Final basis for loc rom assembly."""
        dir = self.bases_path(nr, method, distr)
        return dir / f'basis_{cell:02}.npy'

    def local_basis_dofs_per_vert(self, nr: int, cell: int, method='hapod', distr='normal') -> Path:
        """Dofs per vertex for each cell."""
        dir = self.bases_path(nr, method, distr)
        return dir / f'dofs_per_vert_{cell:02}.npy'

    def rom_error_u(self, nreal: int, num_modes: int, method='hapod', ei=False) -> Path:
        dir = self.method_folder(nreal, method)
        if ei:
            return dir / f'rom_error_u_ei_{num_modes}.npz'
        else:
            return dir / f'rom_error_u_{num_modes}.npz'

    def rom_error_s(self, nreal: int, num_modes: int, method='hapod', ei=False) -> Path:
        dir = self.method_folder(nreal, method)
        if ei:
            return dir / f'rom_error_s_ei_{num_modes}.npz'
        else:
            return dir / f'rom_error_s_{num_modes}.npz'

    def rom_condition(self, nreal: int, num_modes: int, method='hapod', ei=False) -> Path:
        dir = self.method_folder(nreal, method)
        if ei:
            return dir / f'rom_condition_ei_{num_modes}.npy'
        else:
            return dir / f'rom_condition_{num_modes}.npy'

    @property
    def fom_minimization_data(self) -> Path:
        """FOM minimization data."""
        return self.rf / 'fom_minimization_data.out'

    @property
    def rom_minimization_data(self) -> Path:
        """ROM minimization data."""
        return self.rf / 'rom_minimization_data.out'

    def pp_stress(self, method: str) -> dict[str, Path]:
        """Postprocessing of stress at optimal design."""
        folder = self.method_folder(0, method)
        return {'fom': folder / 'stress_fom.xdmf', 'rom': folder / 'stress_rom.xdmf', 'err': folder / 'stress_err.xdmf'}

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
    def fig_projerr(self, k: int) -> Path:
        return self.figures_path / f'fig_projerr_{k:02}.pdf'

    def fig_rom_error(self, method: str, ei: bool) -> Path:
        if ei:
            return self.figures_path / f'rom_error_{method}_ei.pdf'
        else:
            return self.figures_path / f'rom_error_{method}.pdf'

    @property
    def realizations(self) -> Path:
        """Returns realizations that can be used to create ``np.random.SeedSequence``."""
        file = SRC / 'realizations.npy'
        if not file.exists():
            self._generate_realizations(file)
        return file

    def _generate_realizations(self, outpath: Path) -> None:
        seed = np.random.SeedSequence()
        realizations = seed.generate_state(self.num_real)
        np.save(outpath, realizations)

    def log_basis_construction(self, nr: int, method: str, k: int) -> Path:
        return self.logs_path(nr, method) / f'basis_construction_{k:02}.log'

    def log_projerr(self, nr: int, method: str, k: int) -> Path:
        return self.logs_path(nr, method) / f'projerr_{k}.log'

    def log_gfem(self, nr: int, cell: int, method='hapod') -> Path:
        return self.logs_path(nr, method) / f'gfem_{cell:02}.log'

    def log_run_locrom(self, nr: int, method: str, distr: str, ei: bool = False) -> Path:
        dir = self.logs_path(nr, method)
        if ei:
            return dir / f'run_locrom_ei_{distr}.log'
        else:
            return dir / f'run_locrom_{distr}.log'

    def log_validate_rom(self, nr: int, modes: int, method='hapod', distr='normal', ei=True) -> Path:
        dir = self.logs_path(nr, method)
        if ei:
            return dir / f'validate_rom_{modes}_{distr}_with_ei.log'
        else:
            return dir / f'validate_rom_{modes}_{distr}.log'

    @property
    def log_optimization(self) -> Path:
        return self.logs_path(0, 'hapod') / 'optimization.log'

    def hapod_singular_values(self, nr: int, k: int) -> Path:
        """Singular values of final POD for k-th transfer problem."""
        return self.method_folder(nr, 'hapod') / f'singular_values_{k:02}.npy'

    def hapod_neumann_svals(self, nr: int, k: int) -> Path:
        """Singular values of POD of neumann data for k-th transfer problem."""
        return self.method_folder(nr, 'hapod') / f'neumann_singular_values_{k:02}.npy'

    def hapod_info(self, nr: int, k: int) -> Path:
        """Info on HAPOD, final POD."""
        return self.method_folder(nr, 'hapod') / f'info_{k:02}.out'

    def hapod_modes_xdmf(self, nr: int, k: int) -> Path:
        """Modes of the final POD for k-th transfer problem."""
        dir = self.method_folder(nr, 'hapod') / 'pod_modes'
        return dir / f'modes_{k:02}.xdmf'

    def heuristic_modes_xdmf(self, nr: int, k: int) -> Path:
        """Modes computed by heuristic range finder."""
        dir = self.method_folder(nr, 'heuristic') / 'modes'
        return dir / f'modes_{k:02}.xdmf'

    def hapod_modes_npy(self, nr: int, k: int) -> Path:
        """Modes of the final POD for k-th transfer problem."""
        dir = self.method_folder(nr, 'hapod') / 'pod_modes'
        return dir / f'modes_{k:02}.npy'

    def heuristic_modes_npy(self, nr: int, k: int) -> Path:
        """Modes for k-th transfer problem."""
        dir = self.method_folder(nr, 'heuristic') / 'modes'
        return dir / f'modes_{k:02}.npy'

    def heuristic_neumann_svals(self, nr: int, k: int) -> Path:
        """Singular values of POD of neumann snapshots."""
        dir = self.method_folder(nr, 'heuristic')
        return dir / f'neumann_svals_{k:02}.npy'

    def projerr(self, nr: int, method: str, k: int) -> Path:
        dir = self.method_folder(nr, method)
        return dir / f'projerr_{k}.npz'

    def path_omega(self, k: int) -> Path:
        return self.grids_path / f'omega_{k:02}.xdmf'

    def path_omega_coarse(self, k: int) -> Path:
        return self.grids_path / f'omega_coarse_{k:02}.msh'

    def path_omega_in(self, k: int) -> Path:
        return self.grids_path / f'omega_in_{k:02}.xdmf'

    def config_to_omega_in(self, config: str, local=True) -> list[int]:
        """Maps config to cell local index/indices of oversampling domain that correspond to omega in."""
        global_indices = {'left': [0, 1], 'right': [8, 9], 'inner': [4, 5]}
        local_indices = {'left': [0, 1], 'right': [1, 2], 'inner': [1, 2]}
        if local:
            return local_indices[config]
        else:
            return global_indices[config]

    # FIXME: not needed for GFEM, but I may use edge functions as well.
    @property
    def cell_sets(self):
        """Returns cell sets for definition of edge basis configuration."""
        # the order is important
        # this way e.g. cell 1 will load modes for the left edge
        # from basis generated for cell 1 (config inner)
        cell_sets = {
            'inner': set([1, 2, 3, 4, 5, 6, 7, 8]),
            'left': set([0]),
            'right': set([9]),
        }
        return cell_sets

    @property
    # FIXME: is this used anywhere?
    def cell_sets_oversampling(self):
        """Returns cell sets that define oversampling domains."""
        # see preprocessing.py
        cells = {
            'inner': set([3, 4, 5, 6]),
            'left': set([0, 1]),
            'right': set([8, 9]),
        }
        return cells

    def boundaries(self, domain: df.mesh.Mesh):
        """Returns boundaries (Dirichlet, Neumann) of the global domain.

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
            'support_left': (
                int(101),
                plane_at(xmin[0], 'x'),
            ),
            'support_right': (int(102), point_at([xmax[0], xmin[1], xmin[2]])),
            'support_top': (
                int(194),
                within_range([xmin[0], xmax[1], xmin[2]], [a, xmax[1], xmin[2]]),
            ),
        }

    def get_dirichlet(self, domain: df.mesh.Mesh, config: str) -> Optional[list[dict]]:
        # NOTE
        # this only defines markers using `within_range`
        # code needs to use df.mesh.locate_entities_boundary
        boundaries = self.boundaries(domain)
        _, left = boundaries['support_left']
        _, right = boundaries['support_right']

        bcs = []
        zero = df.default_scalar_type(0.0)

        if config == 'left':
            fix_ux = {
                'value': zero,
                'boundary': left,
                'entity_dim': 1,
                'sub': 0,
            }
            bcs.append(fix_ux)
            return bcs
        elif config == 'inner':
            return None
        elif config == 'right':
            fix_uy = {
                'value': zero,
                'boundary': right,
                'entity_dim': 0,
                'sub': 1,
            }
            bcs.append(fix_uy)
            return bcs
        else:
            raise NotImplementedError

    def get_neumann(self, domain: df.mesh.Mesh, config: str) -> Optional[tuple[int, Callable]]:
        boundaries = self.boundaries(domain)
        tag, marker = boundaries['support_top']

        if config == 'left':
            return (tag, marker)
        elif config == 'inner':
            return None
        elif config == 'right':
            return None
        else:
            raise NotImplementedError
