import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem

from pymor.parameters.base import Mu, Parameters


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

    def compute_interface_coord(self, mu: Mu) -> np.ndarray:
        """Returns transformed coordinates for each point on the parent interface.

        Note: needs to be implemented by user for desired transformation map Φ(μ)
        Return value should have shape (num_points, num_components).flatten()
        """

        omega = self.problem.domain
        tdim = omega.tdim
        fdim = tdim - 1

        omega.grid.topology.create_connectivity(fdim, 0)
        facet_to_vertex = omega.grid.topology.connectivity(fdim, 0)
        facets_interface = omega.facet_tags.find(self.facet_tags["interface"])
        vertices_interface = []
        for facet in facets_interface:
            verts = facet_to_vertex.links(facet)
            vertices_interface.append(verts)
        vertices_interface = np.unique(vertices_interface)
        # coord of interface points on parent domain
        x_p = mesh.compute_midpoints(omega.grid, 0, vertices_interface)
        x_center = omega.xmin + (omega.xmax - omega.xmin) / 2

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
        dummy_bc = fem.dirichletbc(np.array([0] * gdim, dtype=float), dofs_interface, V)
        self._dofs_interface = dummy_bc._cpp_object.dof_indices()[0]

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
        dofs_interface = self._dofs_interface
        g = self._d
        g.x.array[dofs_interface] = self.compute_interface_coord(mu)
        g.x.scatter_forward()
        bc_interface = fem.dirichletbc(g, self._boundary_dofs)

        p = self.problem
        p.assemble_vector(bcs=[bc_interface])
        solver = p.solver
        solver.solve(p.b, u.vector)


def discretize_auxiliary_problem(mshfile: str, degree: int, param: dict[str, int], gdim: int = 2):
    """Discretizes the auxiliary problem to compute transformation displacement.

    Args:
        mshfile: The parent domain.
        degree: Polynomial degree of geometry interpolation.
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

    ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
    aux = AuxiliaryProblem(problem, ftags, param)
    return aux


def main():
    from .tasks import example
    from dolfinx.io.utils import XDMFFile

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    degree = 1

    # discretize auxiliary problem for reference (parent) domain
    mshfile = example.parent_unit_cell.as_posix()
    param = {"R": 1}
    auxp = discretize_auxiliary_problem(mshfile, degree, param)

    # output function
    d = fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    mu_values = []
    mu_values.append(auxp.parameters.parse([0.1]))
    mu_values.append(auxp.parameters.parse([0.3]))

    for time, mu in enumerate(mu_values):
        auxp.solve(d, mu)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()


if __name__ == "__main__":
    main()
