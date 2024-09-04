"""stress analysis"""

import basix
import ufl
import numpy as np
import dolfinx as df
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from petsc4py import PETSc
from parageom.fom import ParaGeomLinEla


def principal_stress_2d(u: df.fem.Function, parageom: ParaGeomLinEla, q_degree: int, cells: np.ndarray, values: np.ndarray):
    """Computes principal Cauchy stress.

    Args:
        u: The displacement field.
        parageom: Geometrically parametrized linear problem.
        q_degree: The quadrature degree.
        cells: The cells for which to evaluate.
        values: Array to fill with values. 

    """

    # Note that class ParaGeomLinEla has transformation displacement d
    # as attribute. d is automatically updated for each call to fom.solve(mu).

    # Mesh
    mesh = u.function_space.mesh

    # Quadrature space and Function for Cauchy stress
    basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)

    # UFL and fem.Expression
    σ = parageom.weighted_stress(u) # type: ignore
    sigma_voigt = ufl.as_vector([σ[0, 0], σ[1, 1], σ[2, 2], σ[0, 1]])
    stress_expr = df.fem.Expression(sigma_voigt, q_points)

    # Compute Cauchy stress
    stress_expr.eval(mesh, entities=cells, values=values)

    def compute_principal_components(f):
        # FIXME
        # how to avoid hardcoding reshape?
        values = f.reshape(cells.size, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx+fyy) / 2 - np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        fmax = (fxx+fyy) / 2 + np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        return fmin, fmax

    # Compute principal stress
    smin, smax = compute_principal_components(values)
    return smin, smax


# credit
# https://github.com/fenics-dolfiny/dolfiny/blob/main/src/dolfiny/projection.py
def project(e, target_func, bcs=[]):
    """Project UFL expression.

    Note
    ----
    This method solves a linear system (using KSP defaults).

    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    dx = ufl.dx(V.mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    v = ufl.TrialFunction(V)
    a = df.fem.form(ufl.inner(v, w) * dx)
    L = df.fem.form(ufl.inner(e, w) * dx)

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    solver.rtol = 1.0e-05
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()
