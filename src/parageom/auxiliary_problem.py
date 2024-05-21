from typing import Union
import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem

from pymor.parameters.base import Mu, Parameters


class GlobalAuxiliaryProblem:
    """Represents auxiliary problem on global parent domain."""

    def __init__(self, problem: LinearElasticityProblem, interface_tags: list[int], parameters: dict[str, int]):
        """Initializes the auxiliary problem.

        Args:
            problem: A linear elastic problem.
            interface_tags: The facet tags for each subdomain interface.
            parameters: Dictionary mapping parameter names to parameter dimensions.

        """

        self.problem = problem
        self.interface_tags = interface_tags
        self.parameters = Parameters(parameters)
        self._init_boundary_dofs(interface_tags)
        self._discretize_lhs()
        # function used to define Dirichlet data on the interface
        self._d = fem.Function(problem.V)  # d = X^μ - X^p
        self._xdofs = self.problem.V.tabulate_dof_coordinates()

    def _discretize_lhs(self):
        petsc_options = None  # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = fem.Constant(
            p.domain.grid, (default_scalar_type(0.0),) * p.domain.gdim
        )
        bc_zero = fem.dirichletbc(u_zero, self._boundary_dofs, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def _init_boundary_dofs(self, interface_tags: list[int]):
        """Initializes dofs on ∂Ω and ∂Ω_int"""

        omega = self.problem.domain
        tdim = omega.tdim
        fdim = tdim - 1
        V = self.problem.V

        alldofs = []
        interface_dofs = {}

        # first determine dofs for each interface
        for k, tag in enumerate(interface_tags):
            dofs = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(tag))
            interface_dofs[k] = dofs
            alldofs.append(dofs)

        # second, determine dofs for ∂Ω
        for boundary in omega.boundaries:
            boundary_locator = omega.str_to_marker(boundary)
            dofs = fem.locate_dofs_geometrical(V, boundary_locator)
            alldofs.append(dofs)

        self._boundary_dofs = np.unique(np.hstack(alldofs)) # union of dofs on ∂Ω and ∂Ω_int

        # this is needed for each subdomain separately, use dictionary
        gdim = omega.gdim
        self._dofs_interface = interface_dofs
        self._dofs_interface_blocked = {}
        for k, idofs in interface_dofs.items():
            dummy_bc = fem.dirichletbc(np.array([0] * gdim, dtype=float), idofs, V)
            self._dofs_interface_blocked[k] = dummy_bc._cpp_object.dof_indices()[0]

    def compute_interface_coord(self, k: int, mu_i: float) -> np.ndarray:
        """Returns transformed coordinates for each point on the parent interface for one of the subdomains.

        Note: needs to be implemented by user for desired transformation map Φ(μ)
        Return value should have shape (num_points, num_components).flatten()
        """

        # center of the subdomain circle
        # good solution: work with the coarse grid
        # quick and dirty: I know the geometry
        UNIT_LENGTH = 1e3 # [mm]
        _xmin = 0.0 + UNIT_LENGTH * k
        _xmax = 1.0 + UNIT_LENGTH * k
        _ymin = 0.0
        _ymax = 1.0
        xmin = np.array([_xmin, _ymin, 0.0])
        xmax = np.array([_xmax, _ymax, 0.0])
        x_center = xmin + (xmax - xmin) / 2

        dofs = self._dofs_interface[k]
        x_p = self._xdofs[dofs]

        x_circle = x_p - x_center
        theta = np.arctan2(x_circle[:, 1], x_circle[:, 0])

        radius = mu_i
        x_new = np.zeros_like(x_p)
        x_new[:, 0] = radius * np.cos(theta)
        x_new[:, 1] = radius * np.sin(theta)
        x_new += x_center
        d_values = x_new - x_p
        return d_values[:, :2].flatten()

    def solve(self, u: fem.Function, mu: Mu) -> None:
        """Solves auxiliary problem.

        Args:
            u: The solution function.
            mu: The parameter value.
        """
        self.parameters.assert_compatible(mu)

        # update parametric mapping
        mu_values = mu.to_numpy()
        g = self._d
        for k, mu_i in enumerate(mu_values):
            dofs_interface = self._dofs_interface_blocked[k]
            g.x.array[dofs_interface] = self.compute_interface_coord(k, mu_i)
        g.x.scatter_forward()
        bc_interface = fem.dirichletbc(g, self._boundary_dofs)

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
        self._d = fem.Function(problem.V)  # d = X^μ - X^p
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
            dofs = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(tag))
            dof_indices[boundary] = dofs
            alldofs.append(dofs)
        self._boundary_dofs = np.unique(np.hstack(alldofs))

        gdim = omega.gdim
        dofs_interface = dof_indices["interface"]
        self._dofs_interface = dofs_interface
        dummy_bc = fem.dirichletbc(np.array([0] * gdim, dtype=float), dofs_interface, V)
        self._dofs_interface_blocked = dummy_bc._cpp_object.dof_indices()[0]

    def _discretize_lhs(self):
        petsc_options = None  # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = fem.Constant(
            p.domain.grid, (default_scalar_type(0.0),) * p.domain.gdim
        )
        bc_zero = fem.dirichletbc(u_zero, self._boundary_dofs, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def solve(self, u: fem.Function, mu: Mu) -> None:
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
        bc_interface = fem.dirichletbc(g, self._boundary_dofs)

        p = self.problem
        p.assemble_vector(bcs=[bc_interface])
        solver = p.solver
        solver.solve(p.b, u.vector)


def discretize_auxiliary_problem(mshfile: str, degree: int, facet_tags: Union[dict[str, int], list[int]], param: dict[str, int], gdim: int = 2):
    """Discretizes the auxiliary problem to compute transformation displacement.

    Args:
        mshfile: The parent domain.
        degree: Polynomial degree of geometry interpolation.
        facet_tags: Tags for all boundaries (AuxiliaryProblem) or several interfaces (GlobalAuxiliaryProblem).
        param: Dictionary mapping parameter names to parameter dimensions.
        gdim: Geometrical dimension of the mesh.

    """
    comm = MPI.COMM_SELF
    domain, ct, ft = gmshio.read_from_msh(mshfile, comm, gdim=gdim)
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)

    # linear elasticity problem
    emod = fem.Constant(omega.grid, default_scalar_type(1.0))
    nu = fem.Constant(omega.grid, default_scalar_type(0.25))
    mat = LinearElasticMaterial(gdim, E=emod, NU=nu)
    ve = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(domain, ve)
    problem = LinearElasticityProblem(omega, V, phases=mat)

    if isinstance(facet_tags, dict):
        aux = AuxiliaryProblem(problem, facet_tags, param)
    elif isinstance(facet_tags, list):
        aux = GlobalAuxiliaryProblem(problem, facet_tags, param)
    return aux


def main():
    from .tasks import example
    from dolfinx.io.utils import XDMFFile

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    degree = example.geom_deg

    # discretize auxiliary problem for parent unit cell
    mshfile = example.parent_unit_cell.as_posix()
    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    param = {"R": 1}
    auxp = discretize_auxiliary_problem(mshfile, degree, ftags, param)

    # output function
    d = fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation_unit_cell.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    mu_values = []
    mu_values.append(auxp.parameters.parse([0.1]))
    mu_values.append(auxp.parameters.parse([0.3]))

    for time, mu in enumerate(mu_values):
        auxp.solve(d, mu)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()

    global_domain_msh = example.global_parent_domain.as_posix()
    param = {"R": 10}
    int_tags = [i for i in range(15, 25)]
    auxp = discretize_auxiliary_problem(global_domain_msh, degree, int_tags, param)

    d = fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation_global_domain.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    mu_values = []
    mu_values.append(auxp.parameters.parse([0.2 for _ in range(10)]))
    values = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.2, 0.3]
    mu_values.append(auxp.parameters.parse(values))

    for time, mu in enumerate(mu_values):
        auxp.solve(d, mu)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()


if __name__ == "__main__":
    main()
