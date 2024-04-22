import typing
import ufl
import numpy as np

from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import create_matrix, assemble_matrix

from pymor.operators.interface import Operator
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.constructions import ZeroOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from multi.domain import Domain
from multi.problems import LinearProblem
from multi.materials import LinearElasticMaterial


class ParaGeomLinEla(LinearProblem):
    """Represents a geometrically parametrized linear elastic problem."""

    def __init__(
        self,
        domain: Domain,
        V: fem.FunctionSpace,
        E: typing.Union[float, fem.Constant],
        NU: typing.Union[float, fem.Constant],
        d: fem.Function,
    ):
        """Initialize linear elastic model with pull back.

        Args:
            domain: The parent domain.
            V: FE space.
            E: Young's modulus.
            NU: Poisson ratio.
            d: parametric transformation displacement field.
        """
        super().__init__(domain, V)
        self.mat = LinearElasticMaterial(gdim=domain.gdim, E=E, NU=NU)
        self.d = d
        self.dx = ufl.Measure("dx", domain=domain.grid)

    def weighted_stress(self, w: ufl.TrialFunction):  # type: ignore
        """Returns weighted stress as UFL form.

        Args:
            w: TrialFunction.

        Note:
            The weighted stress depends on the current value of the
            transformation displacement field.

        """
        lame_1 = self.mat.lambda_1
        lame_2 = self.mat.lambda_2

        gdim = self.domain.gdim
        Id = ufl.Identity(gdim)

        # pull back
        F = Id + ufl.grad(self.d)  # type: ignore
        detF = ufl.det(F)
        Finv = ufl.inv(F)
        FinvT = ufl.inv(F.T)

        i, j, k, l, m, p = ufl.indices(6)
        tetrad_ikml = lame_1 * Id[i, k] * Id[m, l] + lame_2 * (
            Id[i, m] * Id[k, l] + Id[i, l] * Id[k, m]  # type: ignore
        )
        grad_u_ml = w[m].dx(p) * Finv[p, l]  # type: ignore
        sigma_ij = ufl.as_tensor(tetrad_ikml * grad_u_ml * FinvT[k, j] * detF, (i, j))  # type: ignore
        return sigma_ij

    @property
    def form_lhs(self):
        grad_v_ij = ufl.grad(self.test)
        sigma_ij = self.weighted_stress(self.trial)
        return ufl.inner(grad_v_ij, sigma_ij) * self.dx

    @property
    def form_rhs(self):
        v = self.test
        zero = fem.Constant(
            self.domain.grid, (default_scalar_type(0.0), default_scalar_type(0.0))
        )
        rhs = ufl.inner(zero, v) * ufl.dx

        if self._bc_handler.has_neumann:
            rhs += self._bc_handler.neumann_bcs

        return rhs


# TODO type hints and docstring
# TODO params and param_setter, support fem.Constant?
class ParaGeomOperator(Operator):
    """Wraps the FEniCSx bilinear form for a geometrically parametrized linear elastic problem as an |Operator|.

    Parameters
    ----------
    form
        The `Form` object which is assembled to a matrix.
    params
        Dict mapping parameters to coefficients in the `Form`.
        These coefficients can be of type `dolfinx.fem.Constant` or `dolfinx.fem.Function`.
    param_setter
        Method to update coefficients according to new value of mu.
        (Solution of the auxiliary problem for new value of mu).
    bcs
        List of `dolfinx.fem.DirichletBC` objects to be applied.
    form_compiler_options
        FFCX Form compiler options.
    jit_options
        JIT compilation options.
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
        param_setter,
        bcs=None,
        form_compiler_options=None,
        jit_options=None,
        solver_options=None,
        name=None,
    ):
        assert len(form.arguments()) == 2
        self.__auto_init(locals())
        range_space = form.arguments()[0].ufl_function_space()
        self.range = FenicsxVectorSpace(range_space)
        source_space = form.arguments()[1].ufl_function_space()
        self.source = FenicsxVectorSpace(source_space)
        self.parameters_own = {k: len(v) for k, v in params.items()}
        self.bcs = bcs or list()

        self.compiled_form = fem.form(
            form, form_compiler_options=form_compiler_options, jit_options=jit_options
        )
        self.matrix = create_matrix(self.compiled_form)

    def _set_mu(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        # NOTE
        # in this context param setter solves the auxiliary problem to get new d(mu)
        # but user could also assign new value to any fem.Constant in the form
        if self.param_setter is not None:
            self.param_setter(mu)

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        # update coefficients in form
        self._set_mu(mu)
        # assemble matrix
        self.matrix.zeroEntries()
        assemble_matrix(self.matrix, self.compiled_form, bcs=self.bcs)
        self.matrix.assemble()

        return FenicsxMatrixOperator(
            self.matrix,
            self.source.V,
            self.range.V,
            self.solver_options,
            self.name + "_assembled",
        )

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

    # def restricted(self, dofs):
    #     # FIXME
    #     with self.logger.block(f"Restricting operator to {len(dofs)} dofs ..."):
    #         if len(dofs) == 0:
    #             return ZeroOperator(NumpyVectorSpace(0), NumpyVectorSpace(0)), np.array(
    #                 [], dtype=int
    #             )
    #
    #         if self.source.V.mesh.__hash__() != self.range.V.mesh.__hash__():
    #             raise NotImplementedError
    #
    #         def blocked(dofs: np.ndarray, bs: int):
    #             ndofs = dofs.size
    #             blocked = np.zeros((ndofs, bs), dtype=dofs.dtype)
    #             for i in range(bs):
    #                 blocked[:, i] = i
    #             r = blocked + np.repeat(dofs[:, np.newaxis], bs, axis=1) * bs
    #             return r.flatten()
    #
    #         self.logger.info("Computing affected cells ...")
    #         domain = self.source.V.mesh
    #         range_dofmap = self.range.V.dofmap
    #         num_cells = domain.topology.index_map(domain.topology.dim).size_local
    #         affected_cell_indices = set()
    #         for cell_index in range(num_cells):
    #             local_dofs = range_dofmap.cell_dofs(cell_index)
    #             # dofmap.cell_dofs() returns non-blocked dof indices
    #             for ld in blocked(local_dofs, range_dofmap.bs):
    #                 if ld in dofs:
    #                     affected_cell_indices.add(cell_index)
    #                     continue
    #         affected_cell_indices = list(sorted(affected_cell_indices))
    #
    #         if any(
    #             i.integral_type() not in ("cell", "exterior_facet")
    #             for i in self.form.integrals()
    #         ):
    #             # enlarge affected_cell_indices if needed
    #             raise NotImplementedError
    #
    #         self.logger.info("Computing source DOFs ...")
    #         source_dofmap = self.source.V.dofmap
    #         source_dofs = set()
    #         for cell_index in affected_cell_indices:
    #             local_dofs = source_dofmap.cell_dofs(cell_index)
    #             # dofmap.cell_dofs returns non-blocked dof indices
    #             source_dofs.update(blocked(local_dofs, source_dofmap.bs))
    #         source_dofs = np.array(sorted(source_dofs), dtype=np.int32)
    #
    #         self.logger.info("Building submesh ...")
    #         tdim = domain.topology.dim
    #         submesh = mesh.create_submesh(domain, tdim, affected_cell_indices)[0]
    #
    #         self.logger.info("Building UFL form on submesh ...")
    #         form_r, V_r_source, V_r_range, source_function_r = self._restrict_form(
    #             submesh, source_dofs
    #         )
    #
    #         self.logger.info("Building DirichletBCs on submesh ...")
    #         bc_r = self._restrict_dirichlet_bcs(submesh, source_dofs, V_r_source)
    #
    #         self.logger.info("Computing source DOF mapping ...")
    #         restricted_source_dofs = self._build_dof_map(
    #             self.source.V, V_r_source, source_dofs
    #         )
    #
    #         self.logger.info("Computing range DOF mapping ...")
    #         restricted_range_dofs = self._build_dof_map(self.range.V, V_r_range, dofs)
    #
    #         op_r = FenicsOperator(
    #             form_r,
    #             FenicsVectorSpace(V_r_source),
    #             FenicsVectorSpace(V_r_range),
    #             source_function_r,
    #             dirichlet_bcs=bc_r,
    #             parameter_setter=self.parameter_setter,
    #             parameters=self.parameters,
    #         )
    #
    #         return (
    #             RestrictedFenicsOperator(op_r, restricted_range_dofs),
    #             source_dofs[np.argsort(restricted_source_dofs)],
    #         )


    # def _restrict_form(self, submesh, source_dofs):
    #     # TODO
    #     # figure out how to restrict the given form to the submesh
    #
    #     V_r_source = fem.functionspace(submesh, self.source.V.ufl_element())
    #     V_r_range = fem.functionspace(submesh, self.range.V.ufl_element())
    #     Vdim = V_r_source.dofmap.bs * V_r_source.dofmap.index_map.size_local
    #     assert Vdim == len(source_dofs)
    #
    #     if self.source.V != self.range.V:
    #         assert all(arg.ufl_function_space() != self.source.V for arg in self.form.arguments())
    #     args = tuple((fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
    #                   if arg.ufl_function_space() == self.range.V else arg)
    #                  for arg in self.form.arguments())
    #
    #     # FIXME
    #     # transformation displacement is a coefficient of the form
    #     # need to create new restricted function and interpolate the values
    #     # of transformation displacement ...
    #     # possibly for every new parameter mu
    #     if any(isinstance(coeff, df.Function) and coeff != self.source_function for coeff in
    #            self.form.coefficients()):
    #         raise NotImplementedError
    #
    #     source_function_r = fem.Function(V_r_source)
    #     # ufl.replace_integral_domains does not exist anymore
    #     form_r = ufl.replace_integral_domains(
    #         self.form(*args, coefficients={self.source_function: source_function_r}),
    #         submesh.ufl_domain()
    #     )
    #
    #     return form_r, V_r_source, V_r_range, source_function_r

    # def _restrict_dirichlet_bcs(self, submesh, source_dofs, V_r_source):
    #     mesh = self.source.V.mesh()
    #     parent_facet_indices = compute_parent_facet_indices(submesh, mesh)
    #
    #     def restrict_dirichlet_bc(bc):
    #         # ensure that markers are initialized
    #         bc.get_boundary_values()
    #         facets = np.zeros(mesh.num_facets(), dtype=np.uint)
    #         facets[bc.markers()] = 1
    #         facets_r = facets[parent_facet_indices]
    #         sub_domains = df.MeshFunction('size_t', submesh, mesh.topology().dim() - 1)
    #         sub_domains.array()[:] = facets_r
    #
    #         bc_r = df.DirichletBC(V_r_source, bc.value(), sub_domains, 1, bc.method())
    #         return bc_r
    #
    #     return tuple(restrict_dirichlet_bc(bc) for bc in self.dirichlet_bcs)

    # def _build_dof_map(self, V, V_r, dofs):
    #     u = df.Function(V)
    #     u_vec = u.vector()
    #     restricted_dofs = []
    #     for dof in dofs:
    #         u_vec.zero()
    #         u_vec[dof] = 1
    #         u_r = df.interpolate(u, V_r)
    #         u_r = u_r.vector().get_local()
    #         if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
    #             raise NotImplementedError
    #         r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
    #         if not len(r_dof) == 1:
    #             raise NotImplementedError
    #         restricted_dofs.append(r_dof[0])
    #     restricted_dofs = np.array(restricted_dofs, dtype=np.int32)
    #     assert len(set(restricted_dofs)) == len(set(dofs))
    #     return restricted_dofs
