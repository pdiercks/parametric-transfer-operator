import typing
from collections import namedtuple

import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
from basix.ufl import element
from mpi4py import MPI
from multi.boundary import plane_at
from multi.domain import Domain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from petsc4py import PETSc
from pymor.algorithms.pod import pod
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace, FenicsxVisualizer
from pymor.core.base import abstractmethod
from pymor.models.basic import StationaryModel
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu, Parameters
from pymor.reductors.basic import StationaryRBReductor
from pymor.vectorarrays.numpy import NumpyVectorSpace

from parageom.definitions import BeamData

AuxiliaryModelWrapper = namedtuple('AuxiliaryModelWrapper', ['model', 'd', 'reductor'], defaults=(None,))


class GlobalAuxiliaryProblem:
    """Represents auxiliary problem on global parent domain."""

    def __init__(
        self,
        problem: LinearElasticityProblem,
        interface_tags: list[int],
        parameters: dict[str, int],
        coarse_grid: StructuredQuadGrid,
        interface_locators: list[typing.Callable],
    ):
        """Initializes the auxiliary problem.

        Args:
            problem: A linear elastic problem.
            interface_tags: The facet tags for each interior subdomain interface.
            parameters: Dictionary mapping parameter names to parameter dimensions.
            coarse_grid: The coarse grid discretization of the global domain.
            interface_locators: Functions to locate the interfaces between unit cells.

        """
        self.problem = problem
        self.interface_tags = interface_tags
        self.parameters = Parameters(parameters)
        self.coarse_grid = coarse_grid
        self._init_boundary_dofs(interface_tags, interface_locators)
        self._discretize_lhs()
        # function used to define Dirichlet data on the interface
        self._d = df.fem.Function(problem.V)  # d = X^μ - X^p
        self._xdofs = self.problem.V.tabulate_dof_coordinates()

    def _discretize_lhs(self):
        petsc_options = None  # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = df.fem.Constant(p.domain.grid, (df.default_scalar_type(0.0),) * p.domain.gdim)
        bc_zero = df.fem.dirichletbc(u_zero, self._boundary_dofs, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def _init_boundary_dofs(self, interface_tags: list[int], interface_locators: list[typing.Callable]):
        """Initializes dofs on ∂Ω and ∂Ω_int."""
        omega = self.problem.domain
        tdim = omega.tdim
        fdim = tdim - 1
        V = self.problem.V

        alldofs = []
        interface_dofs = {}

        # first determine dofs for each interface ∂Ω_int
        for k, tag in enumerate(interface_tags):
            dofs = df.fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(tag))
            interface_dofs[k] = dofs
            alldofs.append(dofs)

        # second, determine dofs for ∂Ω
        for boundary in omega.boundaries:
            boundary_locator = omega.str_to_marker(boundary)
            dofs = df.fem.locate_dofs_geometrical(V, boundary_locator)
            alldofs.append(dofs)

        # add interfaces between unit cells to _boundary_dofs
        # this way the transformation displacement add these interfaces
        # can be constrained to be zero
        for locator in interface_locators:
            dofs = df.fem.locate_dofs_geometrical(V, locator)
            alldofs.append(dofs)

        self._boundary_dofs = np.unique(np.hstack(alldofs))  # union of dofs on ∂Ω and ∂Ω_int

        # this is needed for each subdomain separately, use dictionary
        gdim = omega.gdim
        self._dofs_interface = interface_dofs
        self._dofs_interface_blocked = {}
        for k, idofs in interface_dofs.items():
            dummy_bc = df.fem.dirichletbc(np.array([0] * gdim, dtype=float), idofs, V)
            self._dofs_interface_blocked[k] = dummy_bc._cpp_object.dof_indices()[0]

    def compute_interface_coord(self, k: int, mu_k: float) -> np.ndarray:
        """Returns transformed coordinates for each point on the parent interface for one of the subdomains.

        Args:
            k: cell (subdomain) index.
            mu_k: Parameter component.

        Note: needs to be implemented by user for desired transformation map Φ(μ)
        Return value should have shape (num_points, num_components).flatten()

        """
        grid = self.coarse_grid
        # computes midpoints of entity
        x_center = grid.get_entity_coordinates(grid.tdim, np.array([k], dtype=np.int32))

        dofs = self._dofs_interface[k]
        x_p = self._xdofs[dofs]

        x_circle = x_p - x_center
        theta = np.arctan2(x_circle[:, 1], x_circle[:, 0])

        radius = mu_k
        x_new = np.zeros_like(x_p)
        x_new[:, 0] = radius * np.cos(theta)
        x_new[:, 1] = radius * np.sin(theta)
        x_new += x_center
        d_values = x_new - x_p
        return d_values[:, :2].flatten()

    def solve(self, u: df.fem.Function, mu: Mu) -> None:
        """Solves auxiliary problem.

        Args:
            u: The solution function.
            mu: The parameter value.

        """
        self.parameters.assert_compatible(mu)

        # update parametric mapping
        mu_values = mu.to_numpy()
        g = self._d
        g.x.petsc_vec.zeroEntries()
        for k, mu_i in enumerate(mu_values):
            dofs_interface = self._dofs_interface_blocked[k]
            g.x.array[dofs_interface] = self.compute_interface_coord(k, mu_i)
        g.x.scatter_forward()
        bc_interface = df.fem.dirichletbc(g, self._boundary_dofs)

        p = self.problem
        p.assemble_vector(bcs=[bc_interface])
        solver = p.solver
        solver.solve(p.b, u.vector)


class AuxiliaryProblem:
    """Represents auxiliary problem to compute transformation displacement."""

    def __init__(self, problem: LinearElasticityProblem, facet_tags: dict[str, int], parameters: dict[str, int]):
        """Initializes the auxiliary problem.

        Args:
            problem: A linear elastic problem.
            facet_tags: Tags for facets marking the boundary and the interface.
            parameters: Dictionary mapping parameter names to parameter dimensions.

        """
        # FIXME
        # I now have problem.omega.facet_tags (type meshtags)
        # and self.facet_tags (type dict)
        # Should the dict representation be added to Domain in general?
        self.problem = problem
        self.facet_tags = facet_tags
        self.parameters = Parameters(parameters)

        self._init_boundary_dofs(facet_tags)
        self._discretize_lhs()
        # function used to define Dirichlet data on the interface
        self._d = df.fem.Function(problem.V)  # d = X^μ - X^p
        self._xdofs = self.problem.V.tabulate_dof_coordinates()

    def compute_interface_coord(self, mu: Mu) -> np.ndarray:
        """Returns transformed coordinates for each point on the parent interface.

        Note: needs to be implemented by user for desired transformation map Φ(μ)
        Return value should have shape (num_points, num_components).flatten()
        """
        omega = self.problem.domain
        x_center = omega.xmin + (omega.xmax - omega.xmin) / 2

        dofs = self._dofs_interface
        x_p = self._xdofs[dofs]

        x_circle = x_p - x_center
        theta = np.arctan2(x_circle[:, 1], x_circle[:, 0])

        radius = mu.to_numpy().item()
        x_new = np.zeros_like(x_p)
        x_new[:, 0] = radius * np.cos(theta)
        x_new[:, 1] = radius * np.sin(theta)
        x_new += x_center
        d_values = x_new - x_p
        return d_values[:, :2].flatten()

    def _init_boundary_dofs(self, facet_tags: dict[str, int]):
        assert 'interface' in facet_tags.keys()

        omega = self.problem.domain
        tdim = omega.tdim
        fdim = tdim - 1
        V = self.problem.V
        alldofs = []
        dof_indices = {}
        for boundary, tag in facet_tags.items():
            dofs = df.fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(tag))
            dof_indices[boundary] = dofs
            alldofs.append(dofs)
        self._boundary_dofs = np.unique(np.hstack(alldofs))

        gdim = omega.gdim
        dofs_interface = dof_indices['interface']
        self._dofs_interface = dofs_interface
        dummy_bc = df.fem.dirichletbc(np.array([0] * gdim, dtype=float), dofs_interface, V)
        self._dofs_interface_blocked = dummy_bc._cpp_object.dof_indices()[0]

    def _discretize_lhs(self):
        petsc_options = None  # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = df.fem.Constant(p.domain.grid, (df.default_scalar_type(0.0),) * p.domain.gdim)
        bc_zero = df.fem.dirichletbc(u_zero, self._boundary_dofs, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def solve(self, u: df.fem.Function, mu: Mu) -> None:
        """Solves auxiliary problem.

        Args:
            u: The solution function.
            mu: The parameter value.

        """
        self.parameters.assert_compatible(mu)

        # update parametric mapping
        dofs_interface = self._dofs_interface_blocked
        g = self._d
        g.x.array[dofs_interface] = self.compute_interface_coord(mu)
        g.x.scatter_forward()
        bc_interface = df.fem.dirichletbc(g, self._boundary_dofs)

        p = self.problem
        p.assemble_vector(bcs=[bc_interface])
        solver = p.solver
        solver.solve(p.b, u.vector)


def discretize_auxiliary_problem(
    example: BeamData,
    omega: Domain,
    facet_tags: typing.Union[dict[str, int], list[int]],
    param: dict[str, int],
    coarse_grid: typing.Optional[StructuredQuadGrid] = None,
):
    """Discretizes the auxiliary problem to compute transformation displacement.

    Args:
        example: The example data class.
        omega: The parent domain.
        facet_tags: Tags for all boundaries (AuxiliaryProblem) or several interfaces (GlobalAuxiliaryProblem).
        param: Dictionary mapping parameter names to parameter dimensions.
        coarse_grid: Optional provide coarse grid for `GlobalAuxiliaryProblem`.

    """
    degree = example.preproc.geom_deg
    gdim = example.gdim

    # linear elasticity problem
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    ve = element('P', omega.grid.basix_cell(), degree, shape=(gdim,))
    V = df.fem.functionspace(omega.grid, ve)
    problem = LinearElasticityProblem(omega, V, phases=mat)

    if isinstance(facet_tags, dict):
        aux = AuxiliaryProblem(problem, facet_tags, param)
    elif isinstance(facet_tags, list):
        assert coarse_grid is not None
        interface_locators = []
        for x_coord in range(1, 10):
            x_coord = float(x_coord)
            interface_locators.append(plane_at(x_coord, 'x'))
        aux = GlobalAuxiliaryProblem(problem, facet_tags, param, coarse_grid, interface_locators=interface_locators)
    return aux


class ParametricDirichletLift(Operator):
    """Represents parameter-dependent Dirichlet lift."""

    linear = True
    adjoint = False
    source = NumpyVectorSpace(1)

    def __init__(
        self,
        range: FenicsxVectorSpace,
        compiled_form: typing.Union[df.fem.Form, list[typing.Any], typing.Any],
        inhom: npt.NDArray[np.int32],
        boundary: npt.NDArray[np.int32],
        parameters: Parameters,
    ):
        """Initializes Dirichlet lift.

        Args:
            range: The range space of the LHS operator.
            compiled_form: The form of the LHS operator.
            inhom: Facet indices of boundary for which non-zero values are computed via `self.compute_dirichlet_values`.
            boundary: All facets comprising the Dirichlet boundary.
            parameters: The Parameters the operator depends on.

        """
        self.range = range
        self.compiled_form = compiled_form
        self.inhom = inhom
        self.boundary = boundary
        self.parameters = Parameters(parameters)

        # private members used to define BC objects and update/set values
        # via `apply_lifting` and `set_bc`
        tdim = range.V.mesh.topology.dim
        fdim = tdim - 1
        self._bc_dofs = df.fem.locate_dofs_topological(range.V, fdim, boundary)
        self._g = df.fem.Function(range.V)
        self._bcs = [df.fem.dirichletbc(self._g, self._bc_dofs)]
        self._x = df.la.create_petsc_vector(range.V.dofmap.index_map, range.V.dofmap.bs)

        # TODO: rather loop over gdim
        self._dofs_values = np.sort(
            np.hstack(
                [
                    df.fem.locate_dofs_topological(range.V.sub(0), fdim, inhom),
                    df.fem.locate_dofs_topological(range.V.sub(1), fdim, inhom),
                ]
            )
        )
        self._dofs = df.fem.locate_dofs_topological(range.V, fdim, inhom)

    @abstractmethod
    def compute_dirichlet_values(self, mu=None):
        """Compute the (non-zero) Dirichlet function values."""
        pass

    def _update_dirichlet_function(self, mu=None):
        values = self.compute_dirichlet_values(mu)
        self._g.x.petsc_vec.zeroEntries()
        self._g.x.array[self._dofs_values] = values
        self._g.x.scatter_forward()

    def apply(self, U, mu=None):
        assert U in self.source
        return self.assemble(mu).lincomb(U.to_numpy())

    def as_range_array(self, mu=None):
        return self.assemble(mu)

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        self._update_dirichlet_function(mu)
        self._x.zeroEntries()
        dolfinx.fem.petsc.apply_lifting(self._x, [self.compiled_form], bcs=[self._bcs])
        self._x.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(self._x, self._bcs)
        return self.range.make_array([self._x])


class ParaCircle(ParametricDirichletLift):
    def __init__(
        self,
        range: FenicsxVectorSpace,
        compiled_form: typing.Union[df.fem.Form, list[typing.Any], typing.Any],
        inhom: npt.NDArray[np.int32],
        boundary: npt.NDArray[np.int32],
        parameters: Parameters,
    ):
        """Initializes ParaCircle.

        Args:
            range: The range space of the LHS operator.
            compiled_form: The form of the LHS operator.
            inhom: Facet indices of boundary for which non-zero values are computed via `self.compute_dirichlet_values`.
            boundary: All facets comprising the Dirichlet boundary.
            parameters: The Parameters the operator depends on.

        """
        super().__init__(range, compiled_form, inhom, boundary, parameters)
        omega = self.range.V.mesh
        xmin = np.amin(omega.geometry.x, axis=0)
        xmax = np.amax(omega.geometry.x, axis=0)
        self.x_center = xmin + (xmax - xmin) / 2

        xdofs = self.range.V.tabulate_dof_coordinates()
        self.x_p = xdofs[self._dofs]
        x_circle = self.x_p - self.x_center
        self.theta = np.arctan2(x_circle[:, 1], x_circle[:, 0])

    def compute_dirichlet_values(self, mu=None):
        """Compute the (non-zero) Dirichlet function values for the circle in the middle."""
        radius = mu.to_numpy().item()
        theta = self.theta
        x_p = self.x_p

        x_new = np.zeros_like(x_p)
        x_new[:, 0] = radius * np.cos(theta)
        x_new[:, 1] = radius * np.sin(theta)
        x_new += self.x_center
        d_values = x_new - x_p
        return d_values[:, :2].flatten()


def discretize_auxiliary_model(example: BeamData, omega: Domain):
    """Build FOM of the auxiliary problem on the unit cell."""
    # define linear elastic problem
    degree = example.preproc.geom_deg
    gdim = example.gdim
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    material = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    V = df.fem.functionspace(omega.grid, ('P', degree, (gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=material)

    form_compiler_options = {}
    jit_options = {}
    a_cpp = df.fem.form(problem.form_lhs, form_compiler_options=form_compiler_options, jit_options=jit_options)
    A = dolfinx.fem.petsc.create_matrix(a_cpp)

    tdim = omega.tdim
    fdim = tdim - 1
    facet_tag_defs = {'bottom': 11, 'left': 12, 'right': 13, 'top': 14, 'interface': 15}
    dof_indices = {}
    boundary_facets = []
    for boundary, tag in facet_tag_defs.items():
        dofs = df.fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(tag))
        dof_indices[boundary] = dofs
        boundary_facets.append(omega.facet_tags.find(tag))
    boundary_dofs = np.unique(np.hstack(list(dof_indices.values())))
    boundary_facets = np.unique(np.hstack(boundary_facets))

    zero = df.fem.Constant(omega.grid, (df.default_scalar_type(0.0),) * gdim)
    bc_zero = df.fem.dirichletbc(zero, boundary_dofs, V)

    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, a_cpp, bcs=[bc_zero])
    A.assemble()

    operator = FenicsxMatrixOperator(A, V, V)
    params = Parameters({'R': 1})
    rhs = ParaCircle(operator.range, a_cpp, omega.facet_tags.find(facet_tag_defs['interface']), boundary_facets, params)
    model = StationaryModel(
        operator,
        rhs,
        products={'energy': operator},
        visualizer=FenicsxVisualizer(operator.source),
        name='AuxiliaryModel',
    )
    return model


def reduce_auxiliary_model(example: BeamData, aux: StationaryModel, ntrain: int):
    """Build ROM of Auxiliary problem for unit cell."""
    ps = aux.parameters.space(example.mu_range)
    training_set = ps.sample_uniformly(ntrain)
    U = aux.solution_space.empty(reserve=ntrain)
    for mu in training_set:
        U.append(aux.solve(mu))

    modes, _ = pod(U, product=aux.operator, rtol=1e-6)
    basis = modes[:1]

    reductor = StationaryRBReductor(aux, RB=basis, product=aux.operator)
    rom = reductor.reduce()
    return rom, reductor


def main():
    """Discretize/reduce auxiliary problem for unit cell. Write singular values."""
    from multi.io import read_mesh

    from parageom.tasks import example

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    assert example.fe_deg == example.geom_deg

    # discretize auxiliary problem for parent unit cell
    mshfile = example.parent_unit_cell
    comm = MPI.COMM_SELF
    domain, ct, ft = read_mesh(mshfile, comm, kwargs={'gdim': example.gdim})
    omega = Domain(domain, cell_tags=ct, facet_tags=ft)

    aux = discretize_auxiliary_model(example, omega)
    aux.disable_logging(True)

    # rom, reductor = reduce_auxiliary_model(example, aux, 21)
    ntrain = 21
    ps = aux.parameters.space(example.mu_range)
    training_set = ps.sample_uniformly(ntrain)
    U = aux.solution_space.empty(reserve=ntrain)
    for mu in training_set:
        U.append(aux.solve(mu))

    # set tolerance such that > 1 modes are returned
    modes, svals = pod(U, product=aux.operator, modes=10, rtol=1e-15)
    basis = modes[:1]

    reductor = StationaryRBReductor(aux, RB=basis, product=aux.operator)
    rom = reductor.reduce()

    ntest = 51
    testing_set = ps.sample_randomly(ntest)
    U_fom = aux.solution_space.empty(reserve=ntest)
    U_rom = aux.solution_space.empty(reserve=ntest)
    for mu in testing_set:
        U_fom.append(aux.solve(mu))
        U_rom.append(reductor.reconstruct(rom.solve(mu)))
    err = U_fom - U_rom
    errn = err.norm(aux.products['energy'])
    rel_err = errn / U_fom.norm(aux.products['energy'])
    if np.sum(rel_err) < 1e-12:
        print('Test passed. Writing singular values ...')
        np.save(example.singular_values_auxiliary_problem, svals)
    else:
        print('Warning, reduced auxiliary problem may be inaccurate ...')

    # Solution of auxiliary problem and update of transformation displacement d
    # d = df.fem.Function(aux.solution_space.V)
    # mu = rom.parameters.parse([0.1])
    # urb = rom.solve(mu)
    # U = reductor.reconstruct(urb)
    # d.x.array[:] = U.to_numpy()[0, :]

    # ftags = {'bottom': 11, 'left': 12, 'right': 13, 'top': 14, 'interface': 15}
    # param = {'R': 1}
    # auxp = discretize_auxiliary_problem(example, omega, ftags, param)

    # output function
    # d = df.fem.Function(auxp.problem.V)
    # xdmf = XDMFFile(d.function_space.mesh.comm, './transformation_unit_cell.xdmf', 'w')
    # xdmf.write_mesh(d.function_space.mesh)

    # mu_values = []
    # a = example.unit_length
    # mu_values.append(auxp.parameters.parse([0.1 * a]))
    # mu_values.append(auxp.parameters.parse([0.3 * a]))

    # for time, mu in enumerate(mu_values):
    #     auxp.solve(d, mu)
    #     xdmf.write_function(d.copy(), t=float(time))
    # xdmf.close()

    # global_domain_msh = example.parent_domain('global')
    # omega_gl = Domain(*read_mesh(global_domain_msh, comm, kwargs={'gdim': example.gdim}))
    # global_coarse_domain_msh = example.coarse_grid('global')
    # coarse_grid = StructuredQuadGrid(read_mesh(global_coarse_domain_msh, comm, kwargs={'gdim': example.gdim})[0])
    # param = {'R': 10}
    # int_tags = [i for i in range(15, 25)]
    # auxp = discretize_auxiliary_problem(example, omega_gl, int_tags, param, coarse_grid=coarse_grid)

    # d = df.fem.Function(auxp.problem.V)
    # xdmf = XDMFFile(d.function_space.mesh.comm, './transformation_global_domain.xdmf', 'w')
    # xdmf.write_mesh(d.function_space.mesh)

    # mu_values = []
    # mu_values.append(auxp.parameters.parse([0.2 * a for _ in range(10)]))
    # values = np.array([0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.2, 0.3]) * a
    # mu_values.append(auxp.parameters.parse(values))

    # for time, mu in enumerate(mu_values):
    #     auxp.solve(d, mu)
    #     xdmf.write_function(d.copy(), t=float(time))
    # xdmf.close()


if __name__ == '__main__':
    main()
