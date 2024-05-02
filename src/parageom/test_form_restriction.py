import numpy as np
import numpy.typing as npt

from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import create_matrix, assemble_matrix
import ufl

import matplotlib.pyplot as plt
from scipy.sparse import csr_array


# from ufl/finiteelement/form.py
# this method not exist anymore for ufl version > 2023.1.1
def replace_integral_domains(form, common_domain):  # TODO: Move elsewhere
    """Given a form and a domain, assign a common integration domain to
    all integrals.
    Does not modify the input form (``Form`` should always be
    immutable).  This is to support ill formed forms with no domain
    specified, sometimes occurring in pydolfin, e.g. assemble(1*dx,
    mesh=mesh).
    """
    domains = form.ufl_domains()
    if common_domain is not None:
        gdim = common_domain.geometric_dimension()
        tdim = common_domain.topological_dimension()
        if not all(
            (
                gdim == domain.geometric_dimension()
                and tdim == domain.topological_dimension()
            )
            for domain in domains
        ):
            raise ValueError(
                "Common domain does not share dimensions with form domains."
            )

    reconstruct = False
    integrals = []
    for itg in form.integrals():
        domain = itg.ufl_domain()
        if domain != common_domain:
            itg = itg.reconstruct(domain=common_domain)
            reconstruct = True
        integrals.append(itg)
    if reconstruct:
        form = ufl.Form(integrals)
    return form


def blocked(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    ndofs = dofs.size
    blocked = np.zeros((ndofs, bs), dtype=dofs.dtype)
    for i in range(bs):
        blocked[:, i] = i
    r = blocked + np.repeat(dofs[:, np.newaxis], bs, axis=1) * bs
    return r.flatten()


def restrict_form(form, S, R, V_r_source, V_r_range, interpolation_data=None):
    """Restrict `form` to submesh.

    Args:
        form: The UFL form to restrict.
        S: Source space.
        R: Range space.
        V_r_source: Source space on submesh.
        V_r_range: Range space on submesh.
    """

    if S != R:
        assert all(arg.ufl_function_space() != S for arg in form.arguments())

    args = tuple(
        (
            fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
            if arg.ufl_function_space() == R
            else arg
        )
        for arg in form.arguments()
    )

    # FIXME
    # restrict_form for FenicsOperator may require different method
    # there, fem.Function other than unknow solution `u` are not allowed?

    new_coeffs = {}
    for function in form.coefficients():
        # replace coefficients (fem.Function)
        name = function.name
        new_coeffs[function] = fem.Function(V_r_source, name=name)
        new_coeffs[function].interpolate(function, nmm_interpolation_data=interpolation_data) 

    # FIXME
    # the original form contains some fem.Function that is parameter dependent
    # it's up to the user to update via code in `param_setter`
    # However, the form restriction is done only once and the connection
    # between `function` and `new_coeffs[function]` is lost.

    # How can `new_coeffs[function]` be updated to new mu?

    # method `param_setter` must be defined before `FenicsxMatrixBasedOperator.restricted`
    # is called.

    # Workaround: call `restrict_form` each time the operator is evaluated
    # if form contains coefficients

    # OTHER APPROACH
    # do not restrict form at all, but work with custom assembler that only
    # loops over `affected_cells`.
    # Implementation of custom assembler is quite involved (technical) though.

    submesh = V_r_source.mesh
    form_r = replace_integral_domains(
            form(*args, coefficients=new_coeffs),
            submesh.ufl_domain()
            )

    return form_r, V_r_source, V_r_range


def assemble_form(form):
    compiled_form = fem.form(form)
    matrix = create_matrix(compiled_form)
    matrix.zeroEntries()
    assemble_matrix(matrix, compiled_form)
    matrix.assemble()
    return matrix


def affected_cells(V, dofs):
    """Returns affected cells.

    Args:
        V: The FE space.
        dofs: Interpolation dofs for restricted evaluation.
    """
    domain = V.mesh
    dofmap = V.dofmap


    affected_cells = set()
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    for cell in range(num_cells):
        cell_dofs = dofmap.cell_dofs(cell)
        for dof in cell_dofs:
            for b in range(dofmap.bs):
                if dof * dofmap.bs + b in dofs:
                    affected_cells.add(cell)
                    continue

    affected_cells = np.array(list(sorted(affected_cells)), dtype=np.int32)
    return affected_cells

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
    from .test_build_dofmap import build_dof_map

    # num_cells = 20
    # domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    # domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    nx = ny = 6
    width = 1. * nx
    height = 1. * ny
    domain = mesh.create_rectangle(MPI.COMM_WORLD, np.array([[0., 0.], [width, height]]), np.array([nx, ny], dtype=np.int32))
    num_cells = domain.topology.index_map(2).size_local
    print(f"{num_cells=}")
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
    print(f"{idofs=}")
    cells = affected_cells(V, idofs)
    print(f"affected cells: {cells}")
    submesh, cell_map, _, _ = mesh.create_submesh(domain, tdim, cells)
    Vsub = fem.functionspace(submesh, V.ufl_element())

    source_dofmap = V.dofmap
    source_dofs = set()
    for cell_index in cells:
        local_dofs = blocked(source_dofmap.cell_dofs(cell_index), source_dofmap.bs)
        source_dofs.update(local_dofs)
    source_dofs = np.array(sorted(source_dofs), dtype=idofs.dtype)

    interpolation_data = None
    if form_str == "function":
        interpolation_data = fem.create_nonmatching_meshes_interpolation_data(
                Vsub.mesh,
                Vsub.element,
                V.mesh)

    rform, _, _ = restrict_form(mass, V, V, Vsub, Vsub, interpolation_data=interpolation_data)

    # FIXME
    # build dof map seems to be wrong!!!
    restricted_source_dofs = build_dof_map(V, cell_map, Vsub, source_dofs)
    restricted_range_dofs = build_dof_map(V, cell_map, Vsub, idofs)

    breakpoint()
    # sanity check for dof mappings
    x_dofs_V = V.tabulate_dof_coordinates()
    x_dofs_V_r = Vsub.tabulate_dof_coordinates()
    diff = x_dofs_V[source_dofs] - x_dofs_V_r[restricted_source_dofs]
    assert np.sum(np.abs(diff)) < 1e-9

    # when are the restricted source dofs needed?
    # when doing U.dofs(source_dofs) the dofs need to be ordered
    # as the dofs on the submesh, i.e. the restricted source dofs.
    # see computation of RG below

    # Be careful when cells/dofs are sorted and when not???

    M = assemble_form(mass)
    R = assemble_form(rform)

    g = fem.Function(V)
    g.x.array[:] = np.random.rand(num_dofs)

    matrix = csr_array(M.getValuesCSR()[::-1])
    reference = matrix.dot(g.x.array[:])[idofs]

    other = csr_array(R.getValuesCSR()[::-1])
    mask = source_dofs[np.argsort(restricted_source_dofs)]
    RG = other.dot(g.x.array[mask])
    result = RG[restricted_range_dofs]

    breakpoint()

    assert np.allclose(reference, result)



if __name__ == "__main__":
    test("mass")
    test("constant")
    test("function")
