import numpy as np

from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
import ufl


def affected_cells(V, dofs):
    """Returns affected cells.

    Args:
        V: The FE space.
        dofs: Interpolation dofs for restricted evaluation.
    """
    domain = V.mesh
    dofmap = V.dofmap


    affected_cells = []
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    for cell in range(num_cells):
        cell_dofs = dofmap.cell_dofs(cell)
        for dof in cell_dofs:
            for b in range(dofmap.bs):
                if dof * dofmap.bs + b in dofs:
                    affected_cells.append(cell)

    affected_cells = np.array(affected_cells, dtype=np.int32)
    return affected_cells


def test(form_str):
    from .custom_assembler import assemble_matrix

    num_cells = 20
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    V = fem.functionspace(domain, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)

    breakpoint()

    match form_str:
        case "mass":
            mass = ufl.inner(u, v) * dx
        case "constant":
            c = fem.Constant(domain, default_scalar_type(2.3))
            mass = ufl.inner(u, v) * c * dx
        case "function":
            f = fem.Function(V)
            f.interpolate(lambda x: 2 * x[0] ** 2)
            mass = ufl.inner(u, v) * f * dx
        case _:
            raise ValueError

    tdim = domain.topology.dim
    source_dofs = np.array([3, 6, 7, 11, 15], dtype=np.int32)
    cells = affected_cells(V, source_dofs)

    compiled_form = fem.form(mass)
    M = fem.assemble_matrix(compiled_form)

    # restricted evaluation
    R = assemble_matrix(domain, mass, active_entities={"cells": cells})
    breakpoint()


if __name__ == "__main__":
    test("mass")
