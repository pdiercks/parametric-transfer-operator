from typing import Union, NamedTuple, Any, Callable, Optional

import numpy as np
import numpy.typing as npt

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


class SubmeshWrapper(NamedTuple):
    mesh: df.mesh.Mesh
    parent_entities: list[int]
    vertex_map: list[int]
    geom_map: list[int]


def _create_dirichlet_bcs(bcs: tuple[Union[BCGeom, BCTopo], ...]) -> list[df.fem.DirichletBC]:
    """Creates list of `df.fem.DirichletBC`.

    Args:
        bcs: The BC specification.
    """
    r = list()
    for nt in bcs:
        space = None
        if isinstance(nt, BCGeom):
            space = nt.V
            dofs = df.fem.locate_dofs_geometrical(space, nt.locator)
        elif isinstance(nt, BCTopo):
            space = nt.V.sub(nt.sub) if nt.sub is not None else nt.V
            dofs = df.fem.locate_dofs_topological(space, nt.entity_dim, nt.entities)
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
def _replace_integral_domains(form, common_domain):
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


def _restrict_form(form, S, R, submesh: SubmeshWrapper, padding=1e-14):
    """Restrict `form` to submesh.

    Args:
        form: The UFL form to restrict.
        S: Source space.
        R: Range space.
        submesh: The submesh for restricted evaluation.
        padding: Padding parameter for creation of interpolation data.
    """

    # FIXME
    # replace !=, == checks with V.element?

    # Note that for some
    # V = fem.functionspace(domain, S.ufl_element())
    # W = fem.functionspace(domain, R.ufl_element())
    # with S and R being the same object (R = S)
    # V == W returns False

    # If S == R returns True, this leads to the result that each argument
    # in `args` below, will be an element of the space R, which may
    # not be desired.
    # Therefore, V_r_range = V_r_source is set if S == R.

    V_r_source = df.fem.functionspace(submesh.mesh, S.ufl_element())
    if S != R:
        assert all(arg.ufl_function_space() != S for arg in form.arguments())
        V_r_range = df.fem.functionspace(submesh.mesh, R.ufl_element())
    else:
        V_r_range = V_r_source

    interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
        V_r_source.mesh, V_r_source.element, S.mesh, padding=padding
    )

    args = tuple(
        (
            df.fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
            if arg.ufl_function_space() == R
            else arg
        )
        for arg in form.arguments()
    )

    new_coeffs = {}
    for function in form.coefficients():
        # replace coefficients (fem.Function)
        name = function.name
        if function.function_space != S:
            raise NotImplementedError("Restriction of coefficients that are not elements of the source space of the full operator is not supported.")
        new_coeffs[function] = df.fem.Function(V_r_source, name=name)
        new_coeffs[function].interpolate(
            function, nmm_interpolation_data=interp_data
        )

    form_r = _replace_integral_domains(
        form(*args, coefficients=new_coeffs), submesh.mesh.ufl_domain()
    )

    return form_r, V_r_source, V_r_range, interp_data


def _build_dof_map(V, V_r, dofs, interp_data) -> npt.NDArray[np.int32]:
    """
    Args:
        V: Full space.
        V_r: restricted space.
        dofs: magic dofs.
    """
    u = df.fem.Function(V)
    u_vec = u.vector

    u_r = df.fem.Function(V_r)
    u_r_vec = u_r.vector

    restricted_dofs = []
    for dof in dofs:
        u_vec.zeroEntries()
        u_vec.array[dof] = 1
        u_r.interpolate(u, nmm_interpolation_data=interp_data)
        u_r_array = u_r_vec.array
        if not np.all(
            np.logical_or(np.abs(u_r_array) < 1e-10, np.abs(u_r_array - 1.0) < 1e-10)
        ):
            raise NotImplementedError
        r_dof = np.where(np.abs(u_r_array - 1.0) < 1e-10)[0]
        if not len(r_dof) == 1:
            raise NotImplementedError
        restricted_dofs.append(r_dof[0])
    restricted_dofs = np.array(restricted_dofs, dtype=np.int32)
    assert len(set(restricted_dofs)) == len(set(dofs))
    return restricted_dofs


def _affected_cells(V, dofs):
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


def _restrict_bc_topo(mesh, submesh_wrapper, bc, g, V_r):

    def locate_dirichlet_entities(mesh: df.mesh.Mesh, submesh_wrapper: SubmeshWrapper, entity_dim: int, dirichlet: npt.NDArray[np.int32]):
        submesh, parent_cells, _, _ = submesh_wrapper
        tdim = submesh.topology.dim

        mesh.topology.create_connectivity(tdim, entity_dim)
        submesh.topology.create_connectivity(tdim, entity_dim)
        parent_c2e = mesh.topology.connectivity(tdim, entity_dim)
        cell2entity = submesh.topology.connectivity(tdim, entity_dim)

        entities = []

        for cell_index in range(submesh.topology.index_map(tdim).size_local):
            parent_ents = parent_c2e.links(parent_cells[cell_index])
            child_ents = cell2entity.links(cell_index)

            for pent, cent in zip(parent_ents, child_ents):
                if pent in dirichlet:
                    entities.append(cent)

        return np.array(entities, dtype=np.int32)


    dim = bc.entity_dim
    tags = locate_dirichlet_entities(mesh, submesh_wrapper, dim, bc.entities)
    return BCTopo(g, tags, dim, V_r, sub=bc.sub)



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
        Tuple of Dirichlet BCs.
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
        bcs: Optional[tuple[Union[BCGeom, BCTopo], ...]] = None,
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
        self._bcs = bcs or tuple()
        self.bcs = _create_dirichlet_bcs(bcs) if bcs is not None else list()
        # FIXME
        # add check that all bcs have correct function_space attribute??
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

    def restricted(self, dofs, padding=1e-14):

        if len(self.form.arguments()) == 1:
            raise NotImplementedError

        # ### compute affected cells
        S = self.source.V
        R = self.range.V
        domain = S.mesh
        cells = _affected_cells(S, dofs)

        # ### compute source dofs based on affected cells
        source_dofmap = S.dofmap
        source_dofs = set()
        for cell_index in cells:
            local_dofs = source_dofmap.cell_dofs(cell_index)
            for ld in local_dofs:
                for b in range(source_dofmap.bs):
                    source_dofs.add(ld * source_dofmap.bs + b)
        source_dofs = np.array(sorted(source_dofs), dtype=dofs.dtype)

        # ### build submesh
        tdim = domain.topology.dim
        submesh = SubmeshWrapper(*df.mesh.create_submesh(
            domain, tdim, cells
        ))

        # ### restrict form to submesh
        restricted_form, V_r_source, V_r_range, interp_data = _restrict_form(self.form, S, R, submesh, padding=padding)

        # ### restrict Dirichlet BCs
        r_bcs = list()
        for nt in self._bcs:
            if isinstance(nt.value, df.fem.Function):
                # interpolate function to submesh
                g = df.fem.Function(V_r_source, name=nt.value.name)
                g.interpolate(nt.value, nmm_interpolation_data=interp_data)
            else:
                # g is of type df.fem.Constant or np.ndarray
                g = nt.value

            if isinstance(nt, BCGeom):
                rbc = BCGeom(g, nt.locator, V_r_source)
            elif isinstance(nt, BCTopo):
                if nt.entity_dim == 1 and tdim == 3:
                    raise NotImplementedError
                rbc = _restrict_bc_topo(S.mesh, submesh, nt, g, V_r_source)
            else:
                raise TypeError
            r_bcs.append(rbc)
        r_bcs = tuple(r_bcs)

        # ### compute dof mapping source
        restricted_source_dofs = _build_dof_map(S, V_r_source, source_dofs, interp_data)

        # ### compute dof mapping range
        restricted_range_dofs = _build_dof_map(R, V_r_range, dofs, interp_data)

        # sanity checks
        assert (
            source_dofs.size
            == V_r_source.dofmap.bs * V_r_source.dofmap.index_map.size_local
        )
        assert restricted_source_dofs.size == source_dofs.size
        assert restricted_range_dofs.size == dofs.size

        # edge case: form has parametric coefficient
        if self.form.coefficients():
            assert self.param_setter is not None
            set_params = self.param_setter

            # FIXME
            # this assumes that every coefficient in restricted form
            # is an element of source (S) and is interpolated to V_r_source
            # This results in PETSc ERROR Segmentation violation for functions of quadrature spaces!

            def param_setter(mu):
                set_params(mu)
                # in addition to original param setter
                # interpolate coefficients
                for r_coeff in restricted_form.coefficients():
                    for coeff in self.form.coefficients():
                        if r_coeff.name == coeff.name:
                            r_coeff.x.array[:] = 0.0
                            r_coeff.interpolate(
                                coeff, nmm_interpolation_data=interp_data
                            )
        else:
            param_setter = None

        op_r = FenicsxMatrixBasedOperator(
            restricted_form,
            self.params,
            param_setter=param_setter,
            bcs=r_bcs,
            functional=self.functional,
            form_compiler_options=self.form_compiler_options,
            jit_options=self.jit_options,
            solver_options=self.solver_options,
        )

        return (
            RestrictedFenicsxMatrixBasedOperator(op_r, restricted_range_dofs),
            source_dofs[np.argsort(restricted_source_dofs)],
        )


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
        assert U in self.source

        #  ### Old version
        # UU = self.op.source.zeros(len(U))
        # for uu, u in zip(UU.vectors, U.to_numpy()):
        #     uu.real_part.impl[:] = np.ascontiguousarray(u)
        # VV = self.op.apply(UU, mu=mu)
        # V = self.range.zeros(len(VV))
        # for v, vv in zip(V.to_numpy(), VV.vectors):
        #     v[:] = vv.real_part.impl[self.restricted_range_dofs]

        # NOTE
        # old version resulted in PETSc error: unassembled Vector

        UU = self.op.source.from_numpy(U.to_numpy())
        VV = self.op.apply(UU, mu=mu)
        V = self.range.make_array(VV.dofs(self.restricted_range_dofs))
        return V


def test(nx, ny, degree, value_shape):
    from mpi4py import MPI
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    domain = df.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    gdim = domain.geometry.dim
    V = df.fem.functionspace(domain, ("P", degree, value_shape))
    ndofs = V.dofmap.bs * V.dofmap.index_map.size_local
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    params = {"R": 1}
    coeff = df.fem.Function(V, name="coeff")
    match value_shape:
        case ():
            form = ufl.inner(u, v) * coeff * ufl.dx

            def param_setter(mu):
                value = mu["R"]
                coeff.interpolate(lambda x: x[0] * value)

        case (gdim, ):
            grad_u = ufl.grad(u)
            grad_v = ufl.grad(v)
            F = ufl.grad(coeff) + ufl.Identity(gdim)
            i, j, k = ufl.indices(3)
            form = grad_v[i, j] * grad_u[j, k] * F[k, i] * ufl.dx

            def param_setter(mu):
                value = mu["R"]
                coeff.interpolate(lambda x: (x[0] * value, x[1] * value))

        case _:
            raise NotImplementedError

    def bottom(x):
        return np.isclose(x[1], 0.0)

    if len(value_shape):
        dirichlet_value = (df.default_scalar_type(5.4), ) * value_shape[0]
    else:
        dirichlet_value = df.default_scalar_type(5.4)
    u_D = df.fem.Constant(domain, dirichlet_value)
    bc_geom = BCGeom(u_D, bottom, V)

    operator = FenicsxMatrixBasedOperator(form, params, param_setter=param_setter, bcs=(bc_geom, ))

    magic_dofs = set()
    nmagic = 13
    while len(magic_dofs) < nmagic:
        magic_dofs.add(np.random.randint(0, ndofs))
    magic_dofs = np.array(list(sorted(magic_dofs)), dtype=np.int32)
    r_op, r_source_dofs = operator.restricted(magic_dofs)

    red_coeffs = r_op.op.form.coefficients()
    assert len(red_coeffs) == 1
    assert red_coeffs[0].name == "coeff"

    def compare(mu) -> bool:
        U = operator.source.random(3)
        AU = operator.apply(U, mu)

        U_dofs = NumpyVectorSpace.make_array(U.dofs(r_source_dofs))
        r_AU = r_op.apply(U_dofs, mu=mu)
        # uncomment to see that r_AU changes when mu changes
        # print(r_AU)

        is_zero = np.sum(np.abs(AU.to_numpy()[:, magic_dofs] - r_AU.to_numpy()))
        return is_zero < 1e-9

    mus = operator.parameters.space({"R": (0.1, 10.)}).sample_randomly(2)
    success = []
    for mu in mus:
        success.append(compare(mu))

    if all(success):
        print(f"test passed for {degree=}, {value_shape=}")



def test_simple():
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

    params = {"s": c_scalar}  # , "v": c_vec}
    operator = FenicsxMatrixBasedOperator(poisson, params)

    mu1 = operator.parameters.parse([-99.0])
    op1 = operator.assemble(mu1)
    mat1 = csr_array(op1.matrix.getValuesCSR()[::-1])

    mu2 = operator.parameters.parse([99.0])
    op2 = operator.assemble(mu2)
    mat2 = csr_array(op2.matrix.getValuesCSR()[::-1])
    assert (mat1 + mat2).nnz == 0

    # ### Test 2: Scalar Function, bilinear form
    fun = fem.Function(V, name="f")
    poisson_2 = ufl.inner(ufl.grad(u), ufl.grad(v)) * fun * ufl.dx

    params = {"R": 1}

    def param_setter(mu):
        value = mu["R"]
        fun.interpolate(lambda x: x[0] * value)

    operator = FenicsxMatrixBasedOperator(poisson_2, params, param_setter=param_setter)

    mu1 = operator.parameters.parse([-99.0])
    op1 = operator.assemble(mu1)
    mat1 = csr_array(op1.matrix.getValuesCSR()[::-1])

    mu2 = operator.parameters.parse([99.0])
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

    mu1 = op.parameters.parse([-12.0, -12.0])
    op1 = op.assemble(mu1)
    mat1 = op1.as_range_array().to_numpy()

    mu2 = op.parameters.parse([12.0, 12.0])
    op2 = op.assemble(mu2)
    mat2 = op2.as_range_array().to_numpy()
    assert np.linalg.norm(mat1 + mat2) < 1e-6


if __name__ == "__main__":
    # simple assembly of full operator for
    # 1 bilinear form with scalar valued constant
    # 2 bilinear form with scalar valued function
    # 3 linear form with vector-valued constant
    test_simple()

    # testing of FenicsxMatrixBasedOperator
    # - comparison of FMBO and RFMBO
    # - domain tdim 1d, 2d, 3d
    # - value shape, scalar, vector
    # - args in form: constant, function
    # - form: bilinear, linear
    # - with or without bcs

    # below test uses
    # - unit square
    # - degree 1, 2
    # - value shape (), (2,)
    # - some form with coefficient as element of full source space
    # - definition of single BCGeom
    # - comparison of operator.apply(U, mu) and restricted_op.apply(U, mu)
    test(10, 10, 1, ())
    test(10, 10, 2, ())
    test(10, 10, 1, (2,))
    test(10, 10, 2, (2,))
