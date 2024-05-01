from typing import Union, NamedTuple, Any, Callable, Optional

import numpy as np
import numpy.typing as npt

# TODO
# consider using fem.assemble_vector & friends instead of explicitly importing petsc functions?
import ufl
import dolfinx as df
from dolfinx.fem.petsc import (
    create_matrix,
    assemble_matrix,
    create_vector,
    assemble_vector,
    set_bc,
)
from petsc4py import PETSc

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.interface import Operator
from pymor.operators.constructions import VectorOperator, VectorFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace


class BCGeom(NamedTuple):
    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    locator: Callable
    V: df.fem.FunctionSpace


class BCTopo(NamedTuple):
    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    entities: npt.NDArray[np.int32]
    entity_dim: int
    V: df.fem.FunctionSpace
    sub: Optional[int] = None


def _create_dirichlet_bcs(bcs: list[Union[BCGeom, BCTopo]]) -> list[df.fem.DirichletBC]:
    """Creates list of `df.fem.DirichletBC`.

    Args:
        bcs: The BC specification.
    """
    r = list()
    for nt in bcs:
        space = None
        if isinstance(nt, BCGeom):
            space = nt.V
            dofs = df.fem.locate_dofs_geometrical(nt.V, nt.locator)
        elif isinstance(nt, BCTopo):
            space = nt.V.sub(nt.sub) if nt.sub is not None else nt.V
            dofs = df.fem.locate_dofs_topological(nt.V, nt.entity_dim, nt.entities)
        else:
            raise TypeError
        try:
            bc = df.fem.dirichletbc(nt.value, dofs, space)
        except TypeError:
            bc = df.fem.dirichletbc(nt.value, dofs)
        r.append(bc)
    return r


# from ufl/finiteelement/form.py
# this method not exist anymore for ufl version > 2023.1.1
def replace_integral_domains(form, common_domain):
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
            df.fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
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
        new_coeffs[function] = df.fem.Function(V_r_source, name=name)
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

    return form_r


def blocked(dofs: npt.NDArray[np.int32], bs: int) -> npt.NDArray[np.int32]:
    ndofs = dofs.size
    blocked = np.zeros((ndofs, bs), dtype=dofs.dtype)
    for i in range(bs):
        blocked[:, i] = i
    r = blocked + np.repeat(dofs[:, np.newaxis], bs, axis=1) * bs
    return r.flatten()


# def unblock(dofs: np.typing.NDArray[np.int32], bs: int) -> np.typing.NDArray[np.int32]:
#     ndofs = dofs.size // bs
#     blocked = np.zeros((ndofs, bs), dtype=dofs.dtype)
#     for i in range(bs):
#         blocked[:, i] = i
#     r = (dofs.reshape(ndofs, bs) - blocked) // bs
#     return r[:, 0].flatten()


def build_dof_map(V, cell_map, V_r, dofs) -> np.ndarray:
    """Computes interpolation dofs of V_r.

    Args:
        V: The function space.
        cell_map: Indices of parent cells.
        V_r: The restricted space.
        dofs: Interpolation DOFs of V.
    """
    assert V.dofmap.bs == V_r.dofmap.bs

    subdomain = V_r.mesh
    tdim = subdomain.topology.dim
    num_cells = subdomain.topology.index_map(tdim).size_local

    parents = []
    children = []

    for cell in range(num_cells):
        parent_cell = cell_map[cell]
        parent_dofs = blocked(V.dofmap.cell_dofs(parent_cell), V.dofmap.bs)
        child_dofs = blocked(V_r.dofmap.cell_dofs(cell), V_r.dofmap.bs)

        parents.append(parent_dofs)
        children.append(child_dofs)
    parents = np.unique(np.hstack(parents))
    children = np.unique(np.hstack(children))
    indx = np.nonzero(parents[:, np.newaxis] - dofs[np.newaxis, :]==0)[0]
    return children[indx]


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


class FenicsxMatrixBasedOperator(Operator):
    """Wraps a parameterized FEniCSx linear or bilinear form as an |Operator|.

    Parameters
    ----------
    form
        The `ufl.Form` object which is assembled to a matrix or vector.
    params
        Dict mapping parameters to `dolfinx.fem.Constant` or dimension.
    param_setter
        Custom method to update all form coefficients to new parameter value.
        This is required if the form contains parametric `dolfinx.fem.Function`s.
    bcs
        List of Dirichlet BCs.
    functional
        If `True` return a |VectorFunctional| instead of a |VectorOperator| in case
        `form` is a linear form.
    form_compiler_options
        FFCX Form compiler options. See `dolfinx.jit.ffcx_jit`.
    jit_options
        JIT compilation options. See `dolfinx.jit.ffcx_jit`.
    solver_options
        The |solver_options| for the assembled :class:`FenicsxMatrixOperator`.
    name
        Name of the operator.
    """

    linear = True

    def __init__(
        self,
        form: ufl.Form,
        params: dict,
        param_setter: Optional[Callable] = None,
        bcs: Optional[list[Union[BCGeom, BCTopo]]] = None,
        functional: Optional[bool] = False,
        form_compiler_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        solver_options: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        assert 1 <= len(form.arguments()) <= 2
        assert not functional or len(form.arguments()) == 1
        self.__auto_init(locals())
        if len(form.arguments()) == 2 or not functional:
            range_space = form.arguments()[0].ufl_function_space()
            self.range = FenicsxVectorSpace(range_space)
        else:
            self.range = NumpyVectorSpace(1)
        if len(form.arguments()) == 2 or functional:
            source_space = form.arguments()[0 if functional else 1].ufl_function_space()
            self.source = FenicsxVectorSpace(source_space)
        else:
            self.source = NumpyVectorSpace(1)
        parameters_own = {}
        for k, v in params.items():
            try:
                parameters_own[k] = v.value.size
            except AttributeError:
                parameters_own[k] = v
        self.parameters_own = parameters_own
        self._bcs = bcs or list()
        self.bcs = _create_dirichlet_bcs(bcs) if bcs is not None else list()
        self.compiled_form = df.fem.form(
            form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        if len(form.arguments()) == 2:
            self.disc = create_matrix(self.compiled_form)
        else:
            self.disc = create_vector(self.compiled_form)

    def _set_mu(self, mu) -> None:
        assert self.parameters.assert_compatible(mu)
        # 1. if param_setter is None, assume params maps to fem.Constant only
        # 2. if fem.Function's are part of the form and dependent on the parameter, the user needs to provide custom param_setter
        if self.param_setter is None:
            for name, constant in self.params.items():
                constant.value = mu[name]
        else:
            self.param_setter(mu)

    def _assemble_matrix(self):
        self.disc.zeroEntries()
        assemble_matrix(self.disc, self.compiled_form, bcs=self.bcs)
        self.disc.assemble()

    def _assemble_vector(self):
        with self.disc.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self.disc, self.compiled_form)

        # Apply boundary conditions to the rhs
        # FIXME: if self.form is linear and representing the rhs
        # then we would need the compiled form of the lhs in case of inhomogeneous Dirchlet BCs
        # apply_lifting(self.disc, [self.compiled_form_lhs], bcs=[bcs])
        self.disc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.disc, self.bcs)

    def assemble(self, mu=None):
        self._set_mu(mu)

        if len(self.form.arguments()) == 2:
            self._assemble_matrix()
            return FenicsxMatrixOperator(
                self.disc,
                self.source.V,
                self.range.V,
                self.solver_options,
                self.name + "_assembled",
            )
        elif self.functional:
            self._assemble_vector()
            V = self.source.make_array([self.disc])
            return VectorFunctional(V)
        else:
            self._assemble_vector()
            V = self.range.make_array([self.disc])
            return VectorOperator(V)

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(
            V, initial_guess=initial_guess, least_squares=least_squares
        )

    def restricted(self, dofs):
        # FIXME
        # this does not work for linear forms

        # TODO: compute affected cells
        S = self.source.V
        R = self.range.V
        domain = S.mesh
        cells = affected_cells(S, dofs)

        # TODO: compute source dofs based on affected cells
        source_dofmap = S.dofmap
        source_dofs = set()
        for cell_index in cells:
            local_dofs = blocked(source_dofmap.cell_dofs(cell_index), source_dofmap.bs)
            source_dofs.update(local_dofs)
        source_dofs = np.array(sorted(source_dofs), dtype=dofs.dtype)

        # TODO: build submesh
        tdim = domain.topology.dim
        submesh, cell_map, parent_vertex_indices, _ = df.mesh.create_submesh(domain, tdim, cells)

        # TODO: build submesh of dim=fdim for restriction of BCTopo
        fdim = tdim - 1
        submesh.topology.create_connectivity(tdim, fdim)
        c2f = submesh.topology.connectivity(tdim, fdim)
        sub_facets = set()
        for ci in range(submesh.topology.index_map(tdim).size_local):
            facets = c2f.links(ci)
            for fac in facets:
                sub_facets.add(fac)
        sub_facets = np.array(list(sorted(sub_facets)), dtype=np.int32)
        _, parent_facet_indices, _, _ = df.mesh.create_submesh(submesh, fdim, sub_facets)
        # FIXME
        parent_entities = {fdim: parent_facet_indices, fdim-1: parent_vertex_indices}

        # prepare data structures for form restriction
        # cannot create class members after init ...
        V_r_source = df.fem.functionspace(submesh, S.ufl_element())
        V_r_range = df.fem.functionspace(submesh, R.ufl_element())

        interpolation_data = df.fem.create_nonmatching_meshes_interpolation_data(
                V_r_source.mesh,
                V_r_source.element,
                S.mesh)

        # ### restrict form to submesh
        restricted_form = restrict_form(self.form, S, R, V_r_source, V_r_range, interpolation_data=interpolation_data)

        # TODO: support repeated form restriction for evaluation of RestrictedFenicsxMatrixBasedOperator
        # It might be possible to do this only once, if we can find coefficients of form
        # and restricted form, and update according to `param_setter`
        # the RestrictedFenicsxMatrixBasedOperator should handle this

        # TODO: restrict Dirichlet BCs
        # loop over self._bcs and restrict each element of the list
        r_bcs = list()
        for nt in self._bcs:

            if isinstance(nt.value, df.fem.Function):
                # interpolate function to submesh
                g = df.fem.Function(V, name=nt.value.name)
                g.interpolate(nt.value, nmm_interpolation_data=interpolation_data)
            else:
                g = nt.value

            if isinstance(nt, BCGeom):
                rbc = BCGeom(g, nt.locator, V_r_source)
            elif isinstance(nt, BCTopo):
                # map entities to the restricted space / submesh
                # what if V.sub(i) was passed as function space?
                rbc = _restrict_bc_topo(nt, g, V_r_source, parent_entities[nt.entity_dim])
            else:
                raise TypeError
            r_bcs.append(rbc)

        # ### compute dof mapping source
        restricted_source_dofs = build_dof_map(S, cell_map, V_r_source, source_dofs)

        # ### compute dof mapping range
        restricted_range_dofs = build_dof_map(R, cell_map, V_r_range, dofs)

        # sanity checks
        assert source_dofs.size == V_r_source.dofmap.bs * V_r_source.dofmap.index_map.size_local
        assert restricted_source_dofs.size == source_dofs.size
        assert restricted_range_dofs.size == dofs.size

        # edge case: form has parametric coefficient
        if self.form.coefficients():
            assert self.param_setter is not None
            set_params = self.param_setter

            def param_setter(mu):
                set_params(mu)
                # in addition to original param setter
                # interpolate coefficients
                for r_coeff in restricted_form.coefficients():
                    for coeff in self.form.coefficients():
                        if r_coeff.name == coeff.name:
                            r_coeff.interpolate(coeff, nmm_interpolation_data=interpolation_data)
        else:
            param_setter = None

        op_r = FenicsxMatrixBasedOperator(restricted_form, self.params, param_setter=param_setter,
                                          bcs=r_bcs, functional=self.functional, form_compiler_options=self.form_compiler_options,
                                          jit_options=self.jit_options, solver_options=self.solver_options)

        return (RestrictedFenicsxMatrixBasedOperator(op_r, restricted_range_dofs),
                    source_dofs[np.argsort(restricted_source_dofs)])


class RestrictedFenicsxMatrixBasedOperator(Operator):
    """Restricted :class:`FenicsxMatrixBasedOperator`."""

    linear = True

    def __init__(self, op, restricted_range_dofs):
        self.source = NumpyVectorSpace(op.source.dim)
        self.range = NumpyVectorSpace(len(restricted_range_dofs))
        self.op = op
        self.restricted_range_dofs = restricted_range_dofs

    def assemble(self, mu=None):
        operator = self.op.assemble(mu)
        return operator

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)



def _restrict_bc_topo(nt: BCTopo, g, V, parents):
    dim = nt.entity_dim

    num_entities = V.mesh.topology.index_map(dim).size_local
    entities = np.zeros(num_entities, dtype=np.int32)
    entities[nt.entities] = 99
    entities_r = entities[parents]
    tags = np.nonzero(entities_r)[0]
    return BCTopo(g, tags, dim, V)


def test_restriction_mass_no_mu():
    from mpi4py import MPI
    from scipy.sparse import csr_array

    nx = ny = 10
    domain = df.mesh.create_unit_square(MPI.COMM_SELF, nx, ny)
    value_shape = ()
    V = df.fem.functionspace(domain, ("P", 1, value_shape))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    mass = ufl.inner(u, v) * ufl.dx
    op = FenicsxMatrixBasedOperator(mass, {})
    matop = op.assemble()
    M = csr_array(matop.matrix.getValuesCSR()[::-1])

    magic = np.array([0, 5, 8], dtype=np.int32)
    rop, rdofs = op.restricted(magic)
    rmatop = rop.assemble()
    Mred = csr_array(rmatop.matrix.getValuesCSR()[::-1])

    # M at interpolation dofs (magic)
    # should be equal to Mred at rop.restricted_range_dofs
    ref = M[np.ix_(magic, magic)]
    d = rop.restricted_range_dofs
    other = Mred[np.ix_(d, d)]
    breakpoint()

    # FIXME
    # this is not quite right
    assert np.allclose(ref.data, other.data)


def test():
    import numpy as np
    from mpi4py import MPI
    import ufl
    from dolfinx import fem, mesh, default_scalar_type
    from scipy.sparse import csr_array

    domain = mesh.create_unit_interval(MPI.COMM_SELF, 4)
    V = fem.functionspace(domain, ("P", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ### Test 1: Scalar Constant, bilinear form
    c_scalar = fem.Constant(domain, default_scalar_type(0.0))
    poisson = ufl.inner(ufl.grad(u), ufl.grad(v)) * c_scalar * ufl.dx

    params = {"s": c_scalar} #, "v": c_vec}
    operator = FenicsxMatrixBasedOperator(poisson, params)

    mu1 = operator.parameters.parse([-99.])
    op1 = operator.assemble(mu1)
    mat1 = csr_array(op1.matrix.getValuesCSR()[::-1])

    mu2 = operator.parameters.parse([99.])
    op2 = operator.assemble(mu2)
    mat2 = csr_array(op2.matrix.getValuesCSR()[::-1])
    assert (mat1 + mat2).nnz == 0

    # ### Test 2: Scalar Function, bilinear form
    fun = fem.Function(V, name="f")
    poisson_2 = ufl.inner(ufl.grad(u), ufl.grad(v)) * fun * ufl.dx
    # poisson.coefficients() --> list[fem.Function, ...]
    # poisson.constants() --> list[fem.Constant, ...]

    params = {"R": 1}
    def param_setter(mu):
        value = mu["R"]
        fun.interpolate(lambda x: x[0] * value)

    operator = FenicsxMatrixBasedOperator(poisson_2, params, param_setter=param_setter)

    mu1 = operator.parameters.parse([-99.])
    op1 = operator.assemble(mu1)
    mat1 = csr_array(op1.matrix.getValuesCSR()[::-1])

    mu2 = operator.parameters.parse([99.])
    op2 = operator.assemble(mu2)
    mat2 = csr_array(op2.matrix.getValuesCSR()[::-1])
    assert (mat1 + mat2).nnz == 0

    # ### Test 3: Vector Constant, linear form
    square = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    V = fem.functionspace(square, ("P", 1))
    v = ufl.TestFunction(V)
    c_vec = fem.Constant(square, (default_scalar_type(0.0), default_scalar_type(0.0)))
    params = {"c": c_vec}
    form_1 = ufl.inner(ufl.grad(v), c_vec) * ufl.dx
    op = FenicsxMatrixBasedOperator(form_1, params)

    mu1 = op.parameters.parse([-12., -12.])
    op1 = op.assemble(mu1)
    mat1 = op1.as_range_array().to_numpy()

    mu2 = op.parameters.parse([12., 12.])
    op2 = op.assemble(mu2)
    mat2 = op2.as_range_array().to_numpy()
    assert np.linalg.norm(mat1 + mat2) < 1e-6

    # TODO: test with bcs
    # TODO: test restricted evaluation


if __name__ == "__main__":
    # test()
    test_restriction_mass_no_mu()
