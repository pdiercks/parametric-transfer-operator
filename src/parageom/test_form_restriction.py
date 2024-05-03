import numpy as np
import numpy.typing as npt

from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import create_matrix, assemble_matrix
import ufl

import matplotlib.pyplot as plt
from scipy.sparse import csr_array


def assemble_form(form):
    compiled_form = fem.form(form)
    matrix = create_matrix(compiled_form)
    matrix.zeroEntries()
    assemble_matrix(matrix, compiled_form)
    matrix.assemble()
    return matrix


def plot_xdofs(V):

    xdofs = V.tabulate_dof_coordinates()
    gdim = 2
    xdofs = xdofs[:, :gdim]
    x, y = xdofs.T

    plt.figure(1)
    plt.scatter(x, y, facecolors="none", edgecolors="k", marker="o")

    for i, (xx, yy) in enumerate(xdofs):
        plt.annotate(str(i), (xx, yy))


def test(form_str):
    from .matrix_based_operator import _build_dof_map, _affected_cells, _restrict_form

    # num_cells = 20
    # domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    # domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    nx = ny = 12
    width = 1. * nx
    height = 1. * ny
    domain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0., 0.], [width, height]]), np.array([nx, ny], dtype=np.int32))
    # num_cells = domain.topology.index_map(2).size_local
    # print(f"{num_cells=}")
    V = fem.functionspace(domain, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)

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

    # NOTE
    # source dofs = all dofs in the source space
    # magic dofs = only dofs relevant for restricted evaluation
    # overhead = source_dofs - magic_dofs 

    tdim = domain.topology.dim
    num_idofs = 10
    num_dofs = V.dofmap.bs * V.dofmap.index_map.size_local
    idofs = set()
    while len(idofs) < num_idofs:
        idofs.add(np.random.randint(0, num_dofs))
    idofs = np.array(list(sorted(idofs)), dtype=np.int32)
    # print(f"{idofs=}")
    cells = _affected_cells(V, idofs)
    # print(f"affected cells: {cells}")
    submesh, _, _, _ = mesh.create_submesh(domain, tdim, cells)
    Vsub = fem.functionspace(submesh, V.ufl_element())


    source_dofmap = V.dofmap
    source_dofs = set()
    for cell_index in cells:
        local_dofs = source_dofmap.cell_dofs(cell_index)
        for ld in local_dofs:
            for b in range(source_dofmap.bs):
                source_dofs.add(ld * source_dofmap.bs + b)
    source_dofs = np.array(sorted(source_dofs), dtype=idofs.dtype)

    interpolation_data = fem.create_nonmatching_meshes_interpolation_data(
            Vsub.mesh,
            Vsub.element,
            V.mesh)

    rform = _restrict_form(mass, V, V, Vsub, Vsub, interpolation_data=interpolation_data)

    r_source_dofs = _build_dof_map(V, Vsub, source_dofs, interpolation_data)
    r_range_dofs = _build_dof_map(V, Vsub, idofs, interpolation_data)

    # sanity check for dof mappings
    x_dofs_V = V.tabulate_dof_coordinates()
    x_dofs_V_r = Vsub.tabulate_dof_coordinates()
    diff = x_dofs_V[source_dofs[np.argsort(r_source_dofs)]] - x_dofs_V_r
    assert np.sum(np.abs(diff)) < 1e-9

    M = assemble_form(mass)
    R = assemble_form(rform)

    # ### Application of operator and restricted operator to some function g in V
    g = fem.Function(V)
    g.x.array[:] = np.random.rand(num_dofs)

    # full operator
    matrix = csr_array(M.getValuesCSR()[::-1])
    reference = matrix.dot(g.x.array[:])[idofs]

    # restricted operator
    other = csr_array(R.getValuesCSR()[::-1])
    mask = source_dofs[np.argsort(r_source_dofs)]
    RG = other.dot(g.x.array[mask])
    result = RG[r_range_dofs]

    assert np.allclose(reference, result)



if __name__ == "__main__":
    test("mass")
    test("constant")
    test("function")
