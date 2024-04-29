# TODO
# consider using fem.assemble_vector & friends instead of explicitly importing petsc functions?
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
        List of `dolfinx.fem.DirichletBC` to be applied.
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
        form,
        params,
        param_setter=None,
        bcs=None,
        functional=False,
        form_compiler_options=None,
        jit_options=None,
        solver_options=None,
        name=None,
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
        self.bcs = bcs or list()
        self.compiled_form = fem.form(
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


if __name__ == "__main__":
    # test FenicsxMatrixBasedOperator
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
    # poisson.coefficients() --> list[fem.Function, ...]
    # poisson.constants() --> list[fem.Constant, ...]

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
