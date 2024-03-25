import numpy as np

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
from basix.ufl import element

from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem


class AuxiliaryProblem:
    def __init__(self, problem):
        self.problem = problem
        self._init_boundary_dofs()
        self._discretize_lhs()
        # ### Points on the interface
        # create connectivity of facets to vertices
        omega = problem.domain
        tdim = omega.tdim
        fdim = tdim - 1

        omega.grid.topology.create_connectivity(fdim, 0)
        facet_to_vertex = omega.grid.topology.connectivity(fdim, 0)
        vertices_interface = []
        # FIXME do not hardcode tag value?
        for facet in omega.facet_tags.find(15):
            verts = facet_to_vertex.links(facet)
            vertices_interface.append(verts)
        vertices_interface = np.unique(vertices_interface)
        self.x_interface = mesh.compute_midpoints(omega.grid, 0, vertices_interface)
        self.x_center = omega.xmin + (omega.xmax - omega.xmin) / 2
        x_circle = (self.x_interface - self.x_center)
        self.theta = np.arctan2(x_circle[:, 1], x_circle[:, 0])
        self._mapping = fem.Function(problem.V)

    def _init_boundary_dofs(self):
        omega = self.problem.domain
        tdim = omega.tdim
        fdim = tdim - 1
        V = self.problem.V
        _dofs_bottom = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(11))
        _dofs_left = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(12))
        _dofs_right = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(13))
        _dofs_top = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(14))
        _dofs_interface = fem.locate_dofs_topological(V, fdim, omega.facet_tags.find(15))
        self._dofs_all = np.unique(np.hstack((_dofs_bottom, _dofs_left, _dofs_right, _dofs_top, _dofs_interface)))

        gdim = omega.gdim
        dummy_bc = fem.dirichletbc(np.array([0, ] * gdim, dtype=float), _dofs_interface, V)
        self.dofs_interface = dummy_bc._cpp_object.dof_indices()[0]

    def _discretize_lhs(self):
        petsc_options = None # defaults to mumps
        p = self.problem
        p.setup_solver(petsc_options=petsc_options)
        u_zero = fem.Constant(p.domain.grid, (default_scalar_type(0.0), ) * p.domain.gdim)
        bc_zero = fem.dirichletbc(u_zero, self._dofs_all, p.V)
        p.assemble_matrix(bcs=[bc_zero])

    def solve(self, mu, u):
        radius = mu.to_numpy().item()
        x_new = np.zeros_like(self.x_interface)
        x_new[:, 0] = radius * np.cos(self.theta)
        x_new[:, 1] = radius * np.sin(self.theta)
        x_new += self.x_center
        d_value = x_new - self.x_interface

        # update parametric mapping
        dofs_interface = self.dofs_interface
        g = self._mapping
        g.x.array[dofs_interface] = d_value[:, :2].flatten()
        g.x.scatter_forward()
        bc_interface = fem.dirichletbc(g, self._dofs_all)

        rhs = self.problem.b # PETSc.Vec
        self.problem.assemble_vector(bcs=[bc_interface])
        solver = self.problem.solver
        solver.solve(rhs, u.vector)


def discretize_auxiliary_problem(mshfile: str, degree: int):
    comm = MPI.COMM_SELF
    domain, ct, ft = gmshio.read_from_msh(mshfile, comm, gdim=2)
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)

    # linear elasticity problem
    emod = fem.Constant(omega.grid, default_scalar_type(1.0))
    nu = fem.Constant(omega.grid, default_scalar_type(0.25))
    gdim = omega.grid.geometry.dim
    mat = LinearElasticMaterial(gdim, E=emod, NU=nu)
    ve = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(domain, ve)
    problem = LinearElasticityProblem(omega, V, phases=mat)

    aux = AuxiliaryProblem(problem)
    return aux


def main():
    from dolfinx.io.utils import XDMFFile
    from pymor.parameters.base import Parameters
    # define training set

    # transformation displacement is used to construct
    # phyiscal domains/meshes
    # need to use same degree as degree that should be
    # used for the geometry interpolation afterwards
    degree = 1

    # discretize auxiliary problem for μ
    mshfile = "./reference_unit_cell.msh"
    auxp = discretize_auxiliary_problem(mshfile, degree)

    # output function
    d = fem.Function(auxp.problem.V)
    xdmf = XDMFFile(d.function_space.mesh.comm, "./transformation.xdmf", "w")
    xdmf.write_mesh(d.function_space.mesh)

    parameters = Parameters({"R": 1})
    mu_values = []
    mu_values.append(parameters.parse([0.1]))
    mu_values.append(parameters.parse([0.3]))

    for time, mu in enumerate(mu_values):
        auxp.solve(mu, d)
        xdmf.write_function(d.copy(), t=float(time))
    xdmf.close()


if __name__ == "__main__":
    main()
