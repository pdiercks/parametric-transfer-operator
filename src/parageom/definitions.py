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
# target tolerances for validation tasks
ttols_validation = (1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5)


@dataclass
class RomValidation:
    """Input Parameters for ROM validation.

    Args:
        ntest: Size of the validation set.
        num_modes: Number of modes to use.
        seed: Random seed for the validation set.
        fields: Fields for which error is computed.

    """

    ntest: int = 200
    num_modes: tuple[int, ...] = tuple(range(20, 101, 20))
    seed: int = 241690
    fields: tuple[str, str] = ('u', 's')


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
    rrf_ttol: tuple[float, ...] = ttols_validation
    rrf_ftol: float = 1e-15
    rrf_nt: int = 1
    num_enrichments: int = 10
    radius_mu: float = 0.01

    def ntest(self, dim: int):
        """Size of the testing set.

        Args:
            dim: Dimension of the parameter space.

        """
        return 400  # dim * 100

    def ntrain(self, dim: int):
        """Size of the training set.

        Args:
            dim: Dimension of the parameter space.

        """
        return 50  # dim * 30


@dataclass
class HAPOD:
    """Input Parameters for HAPOD.

    Args:
        seed_train: Random seed for training set.
        eps: Bound l2-mean approx. error by this value.
        omega: Trade-off factor.

    """

    seed_train: int = 212854936
    eps: tuple[float, ...] = ttols_validation
    omega: float = 0.5

    def ntrain(self, dim: int):
        """Size of the training set.

        Args:
            dim: Dimension of the parameter space.

        """
        return 400  # dim * 100


@dataclass
class ProjErr:
    """Input Parameters for projection error study.

    Args:
        configs: Which transfer problems to consider.
        seed_test: Random seed for the test set.
        seed_train: Random seed for the training.
        hapod_eps: Bound for HAPOD.
        hapod_omega: Trade-off factor for HAPOD.
        hrrf_tol: Bound for HRRF.

    """

    # Note
    # for comparability set hapod_training_set=hrrf_testing_set
    # see src/parageom/projerr.py

    configs: tuple[int, ...] = (0, 5)
    seed_train: int = 456729121
    seed_test: int = 923719053
    ttol: float = 1e-5
    hapod_omega: float = 0.5


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
    num_modes: int = 100
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
    mdeim_l2err: float = 0.0
    debug: bool = False

    # task parameters
    a = unit_length / characteristic_length
    preproc = PreProcessing(unit_length=a, geom_deg=fe_deg)
    hrrf = HRRF()
    hapod = HAPOD()
    projerr = ProjErr()
    rom_validation = RomValidation()
    opt = Optimization()

    def _scaling(self):
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

    def _make_tree(self):
        # tree
        # work/{self.name} (run folder)
        #                 /grids
        #                 /method
        #                        /realization_{n}
        self.grids.mkdir(exist_ok=True, parents=True)
        self.auxiliary.mkdir(exist_ok=True, parents=True)
        self.figures.mkdir(exist_ok=True, parents=True)
        for method in self.methods:
            for n in range(self.num_real):
                self.logs(method, n).mkdir(exist_ok=True, parents=True)
                self.gfembases(method, n).mkdir(exist_ok=True, parents=True)
                self.modes(method, n).mkdir(exist_ok=True, parents=True)
                self.projection(method, n).mkdir(exist_ok=True, parents=True)
                self.validation(method, n).mkdir(exist_ok=True, parents=True)
                self.optimization(method, n).mkdir(exist_ok=True, parents=True)

    def __post_init__(self):
        """Creates directory structure and dependent attributes."""
        if self.debug:
            self.num_real = 1
        else:
            self.num_real = 20
        self._scaling()
        self._make_tree()

    @property
    def plotting_styles(self) -> dict[str, Path]:
        """PhD thesis mplstyles."""
        return {'thesis': ROOT / 'src/thesis.mplstyle', 'thesis-halfwidth': ROOT / 'src/thesis-halfwidth.mplstyle'}

    @property
    def figures(self) -> Path:
        return ROOT / 'figures' / f'{self.name}'

    @property
    def rf(self) -> Path:
        """Run folder."""
        return WORK / f'{self.name}'

    @property
    def grids(self) -> Path:
        return self.rf / 'grids'

    @property
    def auxiliary(self) -> Path:
        return self.rf / 'auxiliary'

    def method_folder(self, name: str) -> Path:
        """Training strategy / method folder."""
        return self.rf / name

    def real_folder(self, method: str, nr: int) -> Path:
        """Realization folder."""
        return self.rf / method / f'realization_{nr:02}'

    def logs(self, method: str, nr: int) -> Path:
        return self.real_folder(method, nr) / 'logs'

    def gfembases(self, method: str, nr: int) -> Path:
        """Return Path to folder containing local gfem bases.

        Args:
            method: name of the training strategy / method.
            nr: realization index.

        """
        return self.real_folder(method, nr) / 'gfembases'

    def modes(self, method: str, nr: int) -> Path:
        """Return Path to folder containing local bases."""
        return self.real_folder(method, nr) / 'modes'

    def projection(self, method: str, nr: int) -> Path:
        return self.real_folder(method, nr) / 'projection'

    def validation(self, method: str, nr: int) -> Path:
        return self.real_folder(method, nr) / 'validation'

    def optimization(self, method: str, nr: int) -> Path:
        return self.real_folder(method, nr) / 'optimization'

    @property
    def coarse_grid(self) -> Path:
        """Global coarse grid."""
        return self.grids / 'coarse_grid.msh'

    @property
    def fine_grid(self) -> Path:
        return self.grids / 'fine_grid.msh'

    @property
    def parent_unit_cell(self) -> Path:
        return self.grids / 'parent_unit_cell.msh'

    @property
    def singular_values_auxiliary_problem(self) -> Path:
        return self.auxiliary / 'singular_values.npy'

    @property
    def cell_type(self) -> str:
        """The cell type of the parent unit cell mesh."""
        match self.preproc.geom_deg:
            case 1:
                return 'quad'
            case 2:
                return 'quad9'
            case _:
                return 'quad'

    def local_basis_npy(self, nr: int, cell: int, method='hapod') -> Path:
        """Final basis for loc rom assembly."""
        dir = self.gfembases(method, nr)
        return dir / f'basis_{cell:02}.npy'

    def local_basis_dofs_per_vert(self, nr: int, cell: int, method='hapod') -> Path:
        """Dofs per vertex for each cell."""
        dir = self.gfembases(method, nr)
        return dir / f'dofs_per_vert_{cell:02}.npy'

    def rom_error(self, method: str, nreal: int, field: str, num_modes: int, ei: bool) -> Path:
        dir = self.validation(method, nreal)
        _ei = '_ei' if ei else ''
        return dir / f'rom_error_{field}_{num_modes}{_ei}.npz'

    def mean_rom_error(self, method: str, field: str, ei: bool) -> Path:
        dir = self.method_folder(method)
        _ei = '_ei' if ei else ''
        return dir / f'mean_rom_error_{field}{_ei}.npz'

    def rom_condition(self, nreal: int, num_modes: int, method='hapod', ei=False) -> Path:
        dir = self.validation(method, nreal)
        _ei = '_ei' if ei else ''
        return dir / f'rom_condition_{num_modes}{_ei}.npy'

    def fom_minimization_data(self, method: str, nr: int) -> Path:
        """FOM minimization data."""
        return self.optimization(method, nr) / 'fom_minimization_data.out'

    def rom_minimization_data(self, method: str, nr: int) -> Path:
        """ROM minimization data."""
        return self.optimization(method, nr) / 'rom_minimization_data.out'

    def mdeim_data(self) -> Path:
        """MDEIM data."""
        return self.auxiliary / 'mdeim_data.out'

    def fig_mdeim_svals(self) -> Path:
        return self.figures / 'fig_mdeim_svals.pdf'

    def fig_aux_svals(self) -> Path:
        return self.figures / 'fig_aux_svals.pdf'

    def pp_stress(self, method: str, nr: int) -> dict[str, Path]:
        """Postprocessing of stress at optimal design."""
        folder = self.optimization(method, nr)
        return {'fom': folder / 'stress_fom.xdmf', 'rom': folder / 'stress_rom.xdmf', 'err': folder / 'stress_err.xdmf'}

    def fig_projerr(self, k: int, scale: float = 0.1) -> Path:
        return self.figures / f'fig_projerr_{k:02}_scale_{scale}.pdf'

    def fig_max_projerr(self, k: int, scale: float = 0.1) -> Path:
        return self.figures / f'fig_max_projerr_{k:02}_scale_{scale}.pdf'

    def fig_rom_error(self, field: str, ei: bool) -> Path:
        _ei = '_ei' if ei else ''
        return self.figures / f'rom_error_{field}{_ei}.pdf'

    def fig_basis_size(self) -> Path:
        return self.figures / 'basis_size.pdf'

    @property
    def realizations(self) -> Path:
        """Returns realizations that can be used to create ``np.random.SeedSequence``."""
        file = SRC / 'realizations.npy'
        if file.exists():
            n = np.load(file).size
            if n < self.num_real:
                self._generate_realizations(file)
        else:
            self._generate_realizations(file)
        return file

    def _generate_realizations(self, outpath: Path) -> None:
        seed = np.random.SeedSequence()
        realizations = seed.generate_state(self.num_real)
        np.save(outpath, realizations)

    def log_basis_construction(self, nr: int, method: str, k: int) -> Path:
        return self.logs(method, nr) / f'basis_construction_{k:02}.log'

    def log_projerr(self, nr: int, method: str, k: int, scale: float = 0.1) -> Path:
        return self.logs(method, nr) / f'projerr_{k}_scale_{scale}.log'

    def log_gfem(self, nr: int, cell: int, method='hapod') -> Path:
        return self.logs(method, nr) / f'gfem_{cell:02}.log'

    def log_validate_rom(self, nr: int, modes: int, method='hapod', ei=True) -> Path:
        dir = self.logs(method, nr)
        if ei:
            return dir / f'validate_rom_{modes}_with_ei.log'
        else:
            return dir / f'validate_rom_{modes}.log'

    @property
    def log_optimization(self) -> Path:
        return self.logs(self.opt.method, self.opt.nreal) / 'optimization.log'

    def hapod_singular_values(self, nr: int, k: int) -> Path:
        """Singular values of final POD for k-th transfer problem."""
        return self.real_folder('hapod', nr) / f'singular_values_{k:02}.npy'

    def hapod_summary(self, nr: int, k: int) -> Path:
        """Info on HAPOD, final POD."""
        return self.real_folder('hapod', nr) / f'hapod_summary_{k:02}.out'

    def modes_xdmf(self, method: str, nr: int, k: int) -> Path:
        dir = self.modes(method, nr)
        return dir / f'modes_{k:02}.xdmf'

    def modes_npy(self, method: str, nr: int, k: int) -> Path:
        dir = self.modes(method, nr)
        return dir / f'modes_{k:02}.npy'

    def projection_error(self, nr: int, method: str, k: int, scale: float = 0.1) -> Path:
        dir = self.projection(method, nr)
        return dir / f'projerr_{k}_scale_{scale}.npz'

    def mean_projection_error(self, method: str, k: int, scale: float = 0.1) -> Path:
        dir = self.method_folder(method)
        return dir / f'mean_projection_error_{k}_scale_{scale}.npz'

    def path_omega(self, k: int) -> Path:
        return self.grids / f'omega_{k:02}.xdmf'

    def path_omega_coarse(self, k: int) -> Path:
        return self.grids / f'omega_coarse_{k:02}.msh'

    def path_omega_in(self, k: int) -> Path:
        return self.grids / f'omega_in_{k:02}.xdmf'

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
