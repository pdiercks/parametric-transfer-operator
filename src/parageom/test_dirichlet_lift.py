from mpi4py import MPI
import dolfinx as df
import ufl
from multi.boundary import plane_at
import numpy as np
from pymor.bindings.fenicsx import FenicsxVectorSpace


def test():
    from .locmor import DirichletLift
    n = 6
    mesh = df.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    V = df.fem.functionspace(mesh, ("P", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    acpp = df.fem.form(a)

    left = plane_at(0.0, "x")
    tdim = mesh.topology.dim
    facets = df.mesh.locate_entities_boundary(mesh, tdim-1, left)

    lift = DirichletLift(FenicsxVectorSpace(V), acpp, facets)

    A = df.fem.assemble_matrix(acpp)
    mat = A.to_scipy()

    def compare(mat, lift, g):
        Ag = np.dot(-mat[:, lift._dof_indices].todense(), g)
        Ag[:, lift._dof_indices] = g

        other = lift.assemble(g).to_numpy()
        err = Ag - other
        if np.linalg.norm(err) < 1e-9:
            print("pass")

    num_vals = n + 1
    g = np.arange(num_vals, dtype=np.float64)
    compare(mat, lift, g)
    compare(mat, lift, np.random.rand(num_vals))
    compare(mat, lift, np.random.rand(num_vals))

if __name__ == "__main__":
    test()
