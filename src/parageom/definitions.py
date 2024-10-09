"""ParaGeom example definitions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from dolfinx import default_scalar_type
from dolfinx.mesh import Mesh
from multi.boundary import plane_at, point_at, within_range

ROOT = Path(__file__).parents[2]
WORK = ROOT / 'work'
SRC = Path(__file__).parent


@dataclass
class RomValidation:
    """Input Parameters for ROM validation.

    Args:
        ntest: Size of the validation set.
        num_modes: Number of modes to use.
        seed: Random seed for the validation set.

    """

    ntest: int = 200
    num_modes: tuple[int, ...] = tuple(range(20, 81, 20))
    seed: int = 241690


@dataclass
class PreProcessing:
    """Data for preprocessing.

    Args:
        unit_length: Dimensionless unit length.
        geom_deg: Degree of geometry interpolation.
        num_intervals: Number of line elements per edge of the unit cell.

    """

    unit_length: float = 1.0
    geom_deg: int = 2
    num_intervals: int = 12


@dataclass
class HRRF:
    """Input Parameters for HRRF.

    Args:
        seed_train: Random seed for training set.
        seed_test: Random seed for testing set.
        rrf_ttol: Target tolerance.
        rrf_ftol: Failure tolerance.
        rrf_nt: Number of random normal test vectors.
        num_enrichments: Number of samples per enrichment in adaptive LHS.
        radius_mu: Radius in parameter space for the adaptive LHS.

    """

    seed_train: int = 767667058
    seed_test: int = 545445836
    rrf_ttol: float = 0.01
    rrf_ftol: float = 1e-15
    rrf_nt: int = 1
    num_enrichments: int = 10
    radius_mu: float = 0.01

    def ntest(self, dim: int):
        """Size of the testing set.

        Args:
            dim: Dimension of the parameter space.

        """
        return dim * 50

    def ntrain(self, dim: int):
        """Size of the training set.

        Args:
            dim: Dimension of the parameter space.

        """
        return dim * 30


@dataclass
class HAPOD:
    """Input Parameters for HAPOD.

    Args:
        seed_train: Random seed for training set.
        eps: Bound l2-mean approx. error by this value.
        omega: Trade-off factor.

    """

    seed_train: int = 212854936
    eps: float = 0.001
    omega: float = 0.5

    def ntrain(self, dim: int):
        """Size of the training set.

        Args:
            dim: Dimension of the parameter space.

        """
        return dim * 50


@dataclass
class ProjErr:
    """Input Parameters for projection error study.

    Args:
        seed_test: Random seed for the test set.
        seed_train: Random seed for the training.
        hapod_eps: Bound for HAPOD.
        hapod_omega: Trade-off factor for HAPOD.
        hrrf_tol: Bound for HRRF.

    """

    # Note
    # for comparability set hapod_training_set=hrrf_testing_set
    # see src/parageom/projerr.py

    seed_train: int = 456729121
    seed_test: int = 923719053
    hapod_eps: float = 0.001  # scale=0.1
    hapod_omega: float = 0.5
    hrrf_tol: float = 0.01


@dataclass
class Optimization:
    """Input Parameters for optimization problem.

    Args:
        nreal: The realization to use.
        omega: Weigthing factor for mass and compliance in the objective functional.
        method: The method used for basis construction.
        num_modes: Size of local basis.
        minimizer: The minimization algorithm (scipy) to use.

    """

    nreal: int = 0
    omega: float = 0.2
    method: str = 'hrrf'
    num_modes: int = 60
    minimizer: str = 'SLSQP'


@dataclass
class BeamData:
    """Holds example specific parameters and manages filepaths.

    Args:
        name: The name of the example.
        gdim: The geometric dimension of the problem.
        characteristic_length: Scaling factor for coordinates.
        characteristic_displacement: Scaling factor for displacement field.
        unit_length: Unit length of the lattice unit cell.
        nx: Number of coarse grid cells (subdomains) in x.
        ny: Number of coarse grid cells (subdomains) in y.
        fe_deg: Degree of FE field interpolation.
        poisson_ratio: The poisson ratio of the material.
        youngs_modulus: The Young's modulus (reference value) of the material.
        plane_stress: If True, use plane stress assumption.
        traction_y: Value for y-component of traction vector.
        neumann_tag: Tag for the Neumann boundary.
        parameters: Dict of dict mapping parameter name to parameter dimension.
        methods: Methods used for basis construction.
        g_scale: Scale (Amplitude) of random boundary conditions.
        mdeim_rtol: Relative tolerance for the MDEIM approximation.
        debug: Run in debug mode.

    """

    name: str = 'parageom'
    gdim: int = 2
    characteristic_length = 100.0  # [mm]
    characteristic_displacement = 0.1  # [mm]
    unit_length: float = 100.0  # [mm]
    nx: int = 10
    ny: int = 1
    fe_deg: int = 2
    poisson_ratio: float = 0.2
    youngs_modulus: float = 30e3  # [MPa]
    plane_stress: bool = True
    traction_y: float = 0.0375  # [MPa]
    neumann_tag: int = 194
    parameter_name: str = 'R'
    parameter_dim: tuple[int, ...] = (2, 3, 4, 4, 4, 4, 4, 4, 4, 3, 2)
    methods: tuple[str, ...] = ('hapod', 'hrrf')
    g_scale: float = 0.1
    mdeim_rtol: float = 1e-5
    debug: bool = False

    # task parameters
    a = unit_length / characteristic_length
    preproc = PreProcessing(unit_length=a, geom_deg=fe_deg)
    hrrf = HRRF()
    hapod = HAPOD()
    projerr = ProjErr()
    rom_validation = RomValidation()
    opt = Optimization()

    def __post_init__(self):
        """Creates directory structure and dependent attributes."""
        a = self.preproc.unit_length
        self.length: float = a * self.nx
        self.height: float = a * self.ny
        self.mu_range: tuple[float, float] = (0.1 * a, 0.3 * a)
        self.mu_bar: float = 0.2 * a
        E = self.youngs_modulus
        NU = self.poisson_ratio
        self.λ = E * NU / (1 + NU) / (1 - 2 * NU)
        self.μ = E / 2 / (1 + NU)
        # μ_c = 2.4μ, such that E/μ_c = 100
        self.characteristic_mu = 0.024 * self.μ
        self.E = E / self.characteristic_mu
        self.NU = NU
        self.sigma_scale = self.characteristic_length / (self.characteristic_mu * self.characteristic_displacement)
        self.energy_scale = self.characteristic_displacement * np.sqrt(self.characteristic_mu)

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
        else:
            self.num_real = 20

        for nr in range(self.num_real):
            self.real_folder(nr).mkdir(exist_ok=True, parents=True)
            for name in self.methods:
                # self.method_folder(nr, name).mkdir(exist_ok=True, parents=True)
                self.logs_path(nr, name).mkdir(exist_ok=True, parents=True)
                self.bases_path(nr, name).mkdir(exist_ok=True, parents=True)
            (self.method_folder(nr, 'hapod') / 'pod_modes').mkdir(exist_ok=True, parents=True)
            (self.method_folder(nr, 'hrrf') / 'modes').mkdir(exist_ok=True, parents=True)

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

    def bases_path(self, nr: int, method: str) -> Path:
        """Return Path to bases folder.

        Args:
            nr: realization index.
            method: name of the training strategy / method.

        """
        return self.method_folder(nr, method) / 'bases'

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

    def local_basis_npy(self, nr: int, cell: int, method='hapod') -> Path:
        """Final basis for loc rom assembly."""
        dir = self.bases_path(nr, method)
        return dir / f'basis_{cell:02}.npy'

    def local_basis_dofs_per_vert(self, nr: int, cell: int, method='hapod') -> Path:
        """Dofs per vertex for each cell."""
        dir = self.bases_path(nr, method)
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
    def fig_projerr(self, k: int, scale: float = 0.1) -> Path:
        return self.figures_path / f'fig_projerr_{k:02}_scale_{scale}.pdf'

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
        # TODO generate realizations if existing nreal < self.num_real
        return file

    def _generate_realizations(self, outpath: Path) -> None:
        seed = np.random.SeedSequence()
        realizations = seed.generate_state(self.num_real)
        np.save(outpath, realizations)

    def log_basis_construction(self, nr: int, method: str, k: int) -> Path:
        return self.logs_path(nr, method) / f'basis_construction_{k:02}.log'

    def log_projerr(self, nr: int, method: str, k: int, scale: float = 0.1) -> Path:
        return self.logs_path(nr, method) / f'projerr_{k}_scale_{scale}.log'

    def log_gfem(self, nr: int, cell: int, method='hapod') -> Path:
        return self.logs_path(nr, method) / f'gfem_{cell:02}.log'

    def log_validate_rom(self, nr: int, modes: int, method='hapod', ei=True) -> Path:
        dir = self.logs_path(nr, method)
        if ei:
            return dir / f'validate_rom_{modes}_with_ei.log'
        else:
            return dir / f'validate_rom_{modes}.log'

    @property
    def log_optimization(self) -> Path:
        return self.logs_path(self.opt.nreal, self.opt.method) / 'optimization.log'

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
        dir = self.method_folder(nr, 'hrrf') / 'modes'
        return dir / f'modes_{k:02}.xdmf'

    def hapod_modes_npy(self, nr: int, k: int) -> Path:
        """Modes of the final POD for k-th transfer problem."""
        dir = self.method_folder(nr, 'hapod') / 'pod_modes'
        return dir / f'modes_{k:02}.npy'

    def heuristic_modes_npy(self, nr: int, k: int) -> Path:
        """Modes for k-th transfer problem."""
        dir = self.method_folder(nr, 'hrrf') / 'modes'
        return dir / f'modes_{k:02}.npy'

    def heuristic_neumann_svals(self, nr: int, k: int) -> Path:
        """Singular values of POD of neumann snapshots."""
        dir = self.method_folder(nr, 'hrrf')
        return dir / f'neumann_svals_{k:02}.npy'

    def projection_error(self, nr: int, method: str, k: int, scale: float = 0.1) -> Path:
        dir = self.method_folder(nr, method)
        return dir / f'projerr_{k}_scale_{scale}.npz'

    def path_omega(self, k: int) -> Path:
        return self.grids_path / f'omega_{k:02}.xdmf'

    def path_omega_coarse(self, k: int) -> Path:
        return self.grids_path / f'omega_coarse_{k:02}.msh'

    def path_omega_in(self, k: int) -> Path:
        return self.grids_path / f'omega_in_{k:02}.xdmf'

    def boundaries(self, domain: Mesh):
        """Returns boundaries (Dirichlet, Neumann) of the global domain.

        Args:
            domain: The global domain.

        Note:
            This only defines markers and should be used with `df.mesh.locate_entities_boundary`.

        """
        x = domain.geometry.x
        a = self.preproc.unit_length
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

    def get_dirichlet(self, domain: Mesh, config: str) -> Optional[list[dict]]:
        # NOTE
        # this only defines markers using `within_range`
        # code needs to use df.mesh.locate_entities_boundary
        boundaries = self.boundaries(domain)
        _, left = boundaries['support_left']
        _, right = boundaries['support_right']

        bcs = []
        zero = default_scalar_type(0.0)

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

    def get_neumann(self, domain: Mesh, config: str) -> Optional[tuple[int, Callable]]:
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


if __name__ == '__main__':
    parageom = BeamData(name='test-beamdata')
    breakpoint()
