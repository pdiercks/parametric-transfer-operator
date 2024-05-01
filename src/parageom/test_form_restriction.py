import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem.petsc import create_matrix, assemble_matrix
import ufl

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
    from .test_build_dofmap import build_dof_map

    num_cells = 20
    # domain = mesh.create_unit_interval(MPI.COMM_WORLD, num_cells)
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
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
    source_dofs = np.array([3, 6, 7, 11, 15], dtype=np.int32)
    cells = affected_cells(V, source_dofs)
    submesh, cell_map, _, _ = mesh.create_submesh(domain, tdim, np.unique(cells))
    Vsub = fem.functionspace(submesh, V.ufl_element())

    interpolation_data = None
    if form_str == "function":
        interpolation_data = fem.create_nonmatching_meshes_interpolation_data(
                Vsub.mesh,
                Vsub.element,
                V.mesh)

    rform, _, _ = restrict_form(mass, V, V, Vsub, Vsub, interpolation_data=interpolation_data)
    M = assemble_form(mass)
    R = assemble_form(rform)

    matrix = csr_array(M.getValuesCSR()[::-1]).todense()
    other = csr_array(R.getValuesCSR()[::-1]).todense()

    restricted_source_dofs = build_dof_map(V, cell_map, Vsub, source_dofs)
    breakpoint()
    assert np.allclose(matrix[np.ix_(source_dofs, source_dofs)], other[np.ix_(restricted_source_dofs, restricted_source_dofs)])
    breakpoint()

    # FIXME
    # the tests passed for the unit interval, but fail for the unit square
    # for the current configuration using
    # source_dofs[1:]
    # restricted_source_dofs[1:]
    # the test passes

    # make a MWE where you can count dofs, but the test does not pass
    # might be a problem with `build_dof_map`



if __name__ == "__main__":
    test("mass")
    test("constant")
    test("function")
