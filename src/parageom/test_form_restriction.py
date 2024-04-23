import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem.petsc import create_matrix, assemble_matrix
import ufl

from scipy.sparse import csr_array

# from ufl/finiteelement/form.py
# this does not exist anymore for ufl version > 2023.1.1
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
        if not all((gdim == domain.geometric_dimension() and tdim == domain.topological_dimension()) for domain in domains):
            raise ValueError("Common domain does not share dimensions with form domains.")

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


def restrict_form(form, S, R, submesh, source_dofs):
    # TODO
    # figure out how to restrict the given form to the submesh

    V_r_source = fem.functionspace(submesh, S.ufl_element())
    V_r_range = fem.functionspace(submesh, R.ufl_element())
    Vdim = V_r_source.dofmap.bs * V_r_source.dofmap.index_map.size_local
    assert Vdim == len(source_dofs)

    if S != R:
        assert all(arg.ufl_function_space() != S for arg in form.arguments())
    args = tuple((fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
                  if arg.ufl_function_space() == R else arg)
                 for arg in form.arguments())

    # FIXME
    # transformation displacement is a coefficient of the form
    # need to create new restricted function and interpolate the values
    # of transformation displacement ...
    # possibly for every new parameter mu

    # possibility to transfer functions from form to restricted form??

    # TODO
    # def transfer_coefficient_function_from_mesh_to_submesh():

    # on the other hand this code assumes that FenicsOperator is nonlinear and not a bilinear form
    # if any(isinstance(coeff, fem.Function) and coeff != source_function for coeff in
    #        form.coefficients()):
    #     raise NotImplementedError

    # source_function_r = fem.Function(V_r_source)
    # ufl.replace_integral_domains does not exist anymore
    form_r = replace_integral_domains(
        form(*args),
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


def test():
    domain = mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = fem.functionspace(domain, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)
    mass = ufl.inner(u, v) * dx

    tdim = domain.topology.dim
    cells = np.array([4, 5], dtype=np.int32)
    submesh = mesh.create_submesh(domain, tdim, cells)[0]

    source_dofs = np.array([], dtype=np.int32)
    for ci in cells:
        cell_dofs = V.dofmap.cell_dofs(ci)
        source_dofs = np.append(source_dofs, cell_dofs)

    rform, rsource, rrange = restrict_form(mass, V, V, submesh, np.unique(source_dofs))
    M = assemble_form(mass)
    R = assemble_form(rform)
    matrix = csr_array(M.getValuesCSR()[::-1]).todense()
    other = csr_array(R.getValuesCSR()[::-1]).todense()
    # the values in other and matrix[np.ix_(d, d)] with d=np.unique(source_dofs)
    # are equal except for the boundary dofs in the submesh (here its half of the value)
    # so this seems to be correct
    dd = np.unique(source_dofs)
    mat = matrix[np.ix_(dd, dd)]
    err = mat - other
    assert np.isclose(np.sum(err) - mat[0, 0], 1e-9)

    # TODO
    # add test where form contains some Constant
    # add test where form contains some Function


if __name__ == "__main__":
    test()
