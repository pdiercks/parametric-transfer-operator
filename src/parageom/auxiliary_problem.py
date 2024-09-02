from pathlib import Path
from typing import Union, Optional, Callable
import numpy as np

from mpi4py import MPI
import dolfinx as df
from dolfinx.io import gmshio
from basix.ufl import element

from multi.io import read_mesh
from multi.boundary import plane_at
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem

from pymor.parameters.base import Mu, Parameters

from parageom.definitions import BeamData


class GlobalAuxiliaryProblem:
    """Represents auxiliary problem on global parent domain."""

    def __init__(self, problem: LinearElasticityProblem, interface_tags: list[int], parameters: dict[str, int], coarse_grid: StructuredQuadGrid, interface_locators: list[Callable]):
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
        u_zero = df.fem.Constant(
            p.domain.grid, (df.default_scalar_type(0.0),) * p.domain.gdim
        )
        bc_zero = df.fem.dirichletbc(u_zero, self._boundary_dofs, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def _init_boundary_dofs(self, interface_tags: list[int], interface_locators: list[Callable]):
        """Initializes dofs on ∂Ω and ∂Ω_int"""

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

        # TODO
        # add interfaces between unit cells to _boundary_dofs
        # this way the transformation displacement add these interfaces
        # can be constrained to be zero
        for locator in interface_locators:
            dofs = df.fem.locate_dofs_geometrical(V, locator)
            alldofs.append(dofs)

        self._boundary_dofs = np.unique(np.hstack(alldofs)) # union of dofs on ∂Ω and ∂Ω_int

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
        assert "interface" in facet_tags.keys()

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
        dofs_interface = dof_indices["interface"]
        self._dofs_interface = dofs_interface
        dummy_bc = df.fem.dirichletbc(np.array([0] * gdim, dtype=float), dofs_interface, V)
        self._dofs_interface_blocked = dummy_bc._cpp_object.dof_indices()[0]

    def _discretize_lhs(self):
        petsc_options = None  # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = df.fem.Constant(
            p.domain.grid, (df.default_scalar_type(0.0),) * p.domain.gdim
        )
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


def discretize_auxiliary_problem(example: BeamData, fine_grid: str, facet_tags: Union[dict[str, int], list[int]], param: dict[str, int], coarse_grid: Optional[str] = None):
    """Discretizes the auxiliary problem to compute transformation displacement.

    Args:
        example: The example data class.
        fine_grid: The parent domain.
        facet_tags: Tags for all boundaries (AuxiliaryProblem) or several interfaces (GlobalAuxiliaryProblem).
        param: Dictionary mapping parameter names to parameter dimensions.
        coarse_grid: Optional provide coarse grid.

    """
    degree = example.geom_deg
    gdim = example.gdim

    comm = MPI.COMM_SELF
    domain, ct, ft = gmshio.read_from_msh(fine_grid, comm, gdim=gdim)
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)

    # linear elasticity problem
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    ve = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = df.fem.functionspace(domain, ve)
    problem = LinearElasticityProblem(omega, V, phases=mat)

    if isinstance(facet_tags, dict):
        aux = AuxiliaryProblem(problem, facet_tags, param)
    elif isinstance(facet_tags, list):
        assert coarse_grid is not None
        grid, _, _ = read_mesh(Path(coarse_grid), MPI.COMM_SELF, kwargs={"gdim":gdim})
        sgrid = StructuredQuadGrid(grid)
        interface_locators = []
        for x_coord in range(1, 10):
            x_coord = float(x_coord)
            interface_locators.append(plane_at(x_coord, "x"))
        aux = GlobalAuxiliaryProblem(problem, facet_tags, param, sgrid, interface_locators=interface_locators)
    return aux


def main():
    from parageom.tasks import example
    from dolfinx.io.utils import XDMFFile

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    assert example.fe_deg == example.geom_deg

    # discretize auxiliary problem for parent unit cell
    mshfile = example.parent_unit_cell.as_posix()
    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    param = {"R": 1}
    auxp = discretize_auxiliary_problem(example, mshfile, ftags, param)

    # output function
    d = df.fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation_unit_cell.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    mu_values = []
    a = example.unit_length
    mu_values.append(auxp.parameters.parse([0.1 * a]))
    mu_values.append(auxp.parameters.parse([0.3 * a]))

    for time, mu in enumerate(mu_values):
        auxp.solve(d, mu)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()

    global_domain_msh = example.global_parent_domain.as_posix()
    global_coarse_domain = example.coarse_grid("global").as_posix()
    param = {"R": 10}
    int_tags = [i for i in range(15, 25)]
    auxp = discretize_auxiliary_problem(example, global_domain_msh, int_tags, param, coarse_grid=global_coarse_domain)

    d = df.fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation_global_domain.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    mu_values = []
    mu_values.append(auxp.parameters.parse([0.2 * a for _ in range(10)]))
    values = np.array([0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.2, 0.3]) * a
    mu_values.append(auxp.parameters.parse(values))

    for time, mu in enumerate(mu_values):
        auxp.solve(d, mu)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()


if __name__ == "__main__":
    main()
