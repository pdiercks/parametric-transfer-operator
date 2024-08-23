from typing import Tuple, Optional, Union, Any

from collections import defaultdict, namedtuple

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array, csr_array
from scipy.linalg import solve

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as df
import dolfinx.fem.petsc
import ufl

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.operators.constructions import VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Parameters

from multi.boundary import plane_at
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.dofmap import DofMap
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.projection import orthogonal_part
from multi.solver import build_nullspace
from multi.io import read_mesh
from multi.interpolation import make_mapping
from multi.sampling import create_random_values
from multi.utils import LogMixin
from .definitions import BeamData
from .matrix_based_operator import FenicsxMatrixBasedOperator
from .dofmap_gfem import GFEMDofMap


EISubdomainOperatorWrapper = namedtuple(
    "EISubdomainOperator", ["rop", "cb", "interpolation_matrix", "magic_dofs", "m_inv"]
)


class COOMatrixOperator(Operator):
    """Wraps COO matrix data as an |Operator|.

    Args:
        data: COO matrix data. See scipy.sparse.coo_array.
        indexptr: Points to end of data for each cell.
        num_cells: Number of cells.
        shape: The shape of the matrix.
        parameters: The |Parameters| the operator depends on.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    def __init__(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        indexptr: np.ndarray,
        num_cells: int,
        shape: Tuple[int, int],
        parameters: Parameters = {},
        solver_options: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        assert all([d.shape == data[0].shape for d in data])
        self.__auto_init(locals())  # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])
        self._data = data[0].copy()

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)

        data, rows, cols = self.data  # type: ignore
        indexptr = self.indexptr  # type: ignore
        num_cells = self.num_cells  # type: ignore

        new = self._data
        if self.parametric and mu is not None:
            m = mu.to_numpy()
            new[: indexptr[0]] = data[: indexptr[0]] * m[0]
            for i in range(1, num_cells):
                new[indexptr[i - 1] : indexptr[i]] = (
                    data[indexptr[i - 1] : indexptr[i]] * m[i]
                )

        K = coo_array((new, (rows, cols)), shape=self.shape)  # type: ignore
        K.eliminate_zeros()
        return NumpyMatrixOperator(
            K.tocsr(),
            self.source.id,
            self.range.id,
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


class GlobalParaGeomOperator(Operator):
    """Operator for geometrically parametrized linear elastic problem
    in the context of localized MOR.

    Args:
        r_sub_op: Restricted subdomain operator.
        data:
        rows:
        cols:
        indexptr: Points to end of data for each cell.
        num_cells: Number of cells.
        shape: The shape of the matrix.
        parameters: The |Parameters| the operator depends on.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    def __init__(
        self,
        ei_sub_op: EISubdomainOperatorWrapper,
        data: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        indexptr: np.ndarray,
        num_cells: int,
        shape: Tuple[int, int],
        parameters: Parameters = {},
        solver_options: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        self.__auto_init(locals())  # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        assert self.parametric
        assert mu is not None

        op = self.ei_sub_op.rop
        magic_dofs = self.ei_sub_op.magic_dofs
        m_inv = self.ei_sub_op.m_inv
        rdofs = op.restricted_range_dofs[m_inv].reshape(magic_dofs.shape)
        interpolation_matrix = self.ei_sub_op.interpolation_matrix
        indexptr = self.indexptr
        M = len(self.ei_sub_op.cb)

        data = self.data
        new = np.zeros((data.shape[1],), dtype=np.float32)

        for i, mu_i in enumerate(mu.to_numpy()):
            # restricted evaluation of the subdomain operator
            loc_mu = op.parameters.parse([mu_i])
            A = csr_array(op.assemble(loc_mu).matrix.getValuesCSR()[::-1])
            _coeffs = solve(interpolation_matrix, A[rdofs[:, 0], rdofs[:, 1]])
            λ = _coeffs.reshape(1, M)

            subdomain_range = None
            if i == 0:
                subdomain_range = np.s_[0 : indexptr[i]]
            else:
                subdomain_range = np.s_[indexptr[i - 1] : indexptr[i]]
            new[subdomain_range] = np.dot(λ, data[:, subdomain_range])

        K = coo_array((new, (self.rows, self.cols)), shape=self.shape)  # type: ignore
        K.eliminate_zeros()
        return NumpyMatrixOperator(
            K.tocsr(),
            self.source.id,
            self.range.id,
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


def reconstruct(
    U_rb: np.ndarray,
    dofmap: DofMap,
    bases: list[np.ndarray],
    u_local: df.fem.Function,
    u_global: df.fem.Function,
) -> None:
    """Reconstructs rom solution on the global domain.

    Args:
        Urb: ROM solution in the reduced space.
        dofmap: The dofmap of the reduced space.
        bases: Local basis for each subdomain.
        u_local: The local solution field.
        u_global: The global solution field to be filled with values.

    """
    coarse_grid = dofmap.grid
    V = u_global.function_space
    Vsub = u_local.function_space
    submesh = Vsub.mesh
    x_submesh = submesh.geometry.x
    u_global_view = u_global.x.array

    for cell in range(dofmap.num_cells):
        # translate subdomain mesh
        vertices = coarse_grid.get_entities(0, cell)
        dx_cell = coarse_grid.get_entity_coordinates(0, vertices)[0]
        x_submesh += dx_cell

        # fill u_local with rom solution
        basis = bases[cell]
        dofs = dofmap.cell_dofs(cell)

        # fill global field via dof mapping
        V_to_Vsub = make_mapping(Vsub, V)
        u_global_view[V_to_Vsub] = U_rb[0, dofs] @ basis
        u_global.x.scatter_forward()

        # move subdomain mesh to origin
        x_submesh -= dx_cell


def assemble_gfem_system(
    dofmap: GFEMDofMap,
    operator: FenicsxMatrixBasedOperator,
    b: VectorOperator,
    mu,
    bases: list[np.ndarray],
    dofs_per_vert: np.ndarray,
    max_dofs_per_vert: np.ndarray
):
    """Assembles ``operator`` and ``rhs`` for localized ROM as ``StationaryModel``.

    Args:
        dofmap: The dofmap of the global GFEM space.
        operator: Local high fidelity operator for stiffness matrix.
        b: Local high fidelity external force vector.
        mu: Parameter value.
        bases: Local reduced basis for each subdomain.
        dofs_per_vert: Number of active modes per vertex.
        max_dofs_per_vert: Number of maximum modes per vertex.

    Note:
        Without Empirical Interpolation `mu` is required to
        assemble the global operators.

    """

    from .dofmap_gfem import select_modes

    # no need to apply bcs as basis functions should
    # satisfy these automatically
    bc_dofs = np.array([], dtype=np.int32)

    lhs = defaultdict(list)
    rhs = defaultdict(list)
    bc_mat = defaultdict(list)
    local_bases = []

    for ci, mu_i in zip(range(dofmap.num_cells), mu.to_numpy()):
        dofs = dofmap.cell_dofs(ci)

        # assemble full subdomain operator
        loc_mu = operator.parameters.parse([mu_i])
        A = operator.assemble(loc_mu)

        # select active modes
        local_basis = select_modes(bases[ci], dofs_per_vert[ci], max_dofs_per_vert[ci])
        local_bases.append(local_basis)
        B = A.source.from_numpy(local_basis)  # type: ignore

        # local stiffness matrix
        A_local = project(A, B, B)
        element_matrix = A_local.matrix  # type: ignore

        # local external force vector
        if ci == 0:
            b_local = project(b, B, None)
            element_vector = b_local.matrix # type: ignore
        else:
            b_local = A_local.range.zeros(1)
            element_vector = b_local.to_numpy().transpose()

        for ld, x in enumerate(dofs):
            if x in bc_dofs:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(0.0)
            else:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(element_vector[ld, 0])

            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:
                    # Note: in the MOR context set diagonal to zero
                    # for the matrices arising from a_q
                    if x == y:
                        if x not in lhs["diagonals"]:  # only set diagonal entry once
                            lhs["rows"].append(x)
                            lhs["cols"].append(y)
                            lhs["data"].append(0.0)
                            lhs["diagonals"].append(x)
                            bc_mat["rows"].append(x)
                            bc_mat["cols"].append(y)
                            bc_mat["data"].append(1.0)
                            bc_mat["diagonals"].append(x)
                else:
                    lhs["rows"].append(x)
                    lhs["cols"].append(y)
                    lhs["data"].append(element_matrix[ld, k])

        lhs["indexptr"].append(len(lhs["rows"]))
        rhs["indexptr"].append(len(rhs["rows"]))

    Ndofs = dofmap.num_dofs
    data = np.array(lhs["data"])
    rows = np.array(lhs["rows"])
    cols = np.array(lhs["cols"])
    indexptr = np.array(lhs["indexptr"])
    shape = (Ndofs, Ndofs)
    options = None
    op = COOMatrixOperator(
        (data, rows, cols),
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=Parameters({}),
        solver_options=options,
        name="K",
    )

    data = np.array(rhs["data"])
    rows = np.array(rhs["rows"])
    cols = np.array(rhs["cols"])
    indexptr = np.array(rhs["indexptr"])
    shape = (Ndofs, 1)
    rhs_op = COOMatrixOperator(
        (data, rows, cols),
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=Parameters({}),
        solver_options=options,
        name="F",
    )
    return op, rhs_op, local_bases


def assemble_gfem_system_with_ei(
    dofmap: GFEMDofMap,
    ei_sub_op: EISubdomainOperatorWrapper,
    b: VectorOperator,
    bases: list[np.ndarray],
    dofs_per_vert: np.ndarray,
    max_dofs_per_vert: np.ndarray,
    parameters: Parameters,
):
    """Assembles ``operator`` and ``rhs`` for localized ROM as ``StationaryModel``.

    Args:
        dofmap: The dofmap of the global GFEM space.
        ei_sub_op: EISubdomainOperatorWrapper.
        b: Local high fidelity external force vector.
        bases: Local reduced basis for each subdomain / unit cell.
        dofs_per_vert: Number of active modes per vertex.
        max_dofs_per_vert: Number of maximum modes per vertex.
        parameters: The |Parameters| the global ROM depends on.

    """

    from .dofmap_gfem import select_modes

    # no need to apply bcs as basis functions should
    # satisfy these automatically
    bc_dofs = np.array([], dtype=np.int32)

    lhs = defaultdict(list)
    rhs = defaultdict(list)
    bc_mat = defaultdict(list)
    local_bases = []

    cb = ei_sub_op.cb
    source = cb[0].source
    cb_size = len(cb)  # size of collateral basis

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)

        # select active modes
        local_basis = select_modes(bases[ci], dofs_per_vert[ci], max_dofs_per_vert[ci])
        local_bases.append(local_basis)
        B = source.from_numpy(local_basis)  # type: ignore

        # projected matrix operators
        mops_local = [project(A, B, B) for A in cb]
        element_matrices = [a_local.matrix for a_local in mops_local]  # type: ignore

        # projected rhs
        if ci == 0:
            b_local = project(b, B, None)
            element_vector = b_local.matrix  # type: ignore
        else:
            b_local = mops_local[0].range.zeros(1)
            element_vector = b_local.to_numpy().transpose()

        for ld, x in enumerate(dofs):
            if x in bc_dofs:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(0.0)
            else:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(element_vector[ld, 0])

            for k, y in enumerate(dofs):
                if x in bc_dofs or y in bc_dofs:
                    # Note: in the MOR context set diagonal to zero
                    # for the matrices arising from a_q
                    if x == y:
                        if x not in lhs["diagonals"]:  # only set diagonal entry once
                            lhs["rows"].append(x)
                            lhs["cols"].append(y)
                            for m in range(cb_size):
                                lhs[f"data_{m}"].append(0.0)
                            lhs["diagonals"].append(x)
                            bc_mat["rows"].append(x)
                            bc_mat["cols"].append(y)
                            bc_mat["data"].append(1.0)
                            bc_mat["diagonals"].append(x)
                else:
                    lhs["rows"].append(x)
                    lhs["cols"].append(y)
                    for m, elem_mat in enumerate(element_matrices):
                        lhs[f"data_{m}"].append(elem_mat[ld, k])

        lhs["indexptr"].append(len(lhs["rows"]))
        rhs["indexptr"].append(len(rhs["rows"]))

    Ndofs = dofmap.num_dofs

    _data = []
    for m in range(cb_size):
        _data.append(lhs[f"data_{m}"])
    data = np.vstack(_data)
    # stack matrix data as row vectors
    # need to form linear combination with interpolation coeff per row
    # need to account for different geometry (μ) per column using indexptr

    # mu_0 will give coefficient vector of length M
    # and each entries in data corresponding to subdomain 0 need to
    # be multiplied with the coefficients and summed up

    # The summation is finally handled by the COO array
    # although here I need data with shape (1, nnz)

    # After solving the interpolation eq. for each subdomain
    # all coefficients can be stored in a matrix of shape (M, 10)

    # data (M, nnz)
    # coeff (M, 10)
    # indexptr --> defining 10 ranges within [0, nnz-1] that corresponds to values for each subdomain

    rows = np.array(lhs["rows"])
    cols = np.array(lhs["cols"])
    indexptr = np.array(lhs["indexptr"])
    shape = (Ndofs, Ndofs)
    options = {"inverse": "scipy_lgmres"}
    op = GlobalParaGeomOperator(
        ei_sub_op,
        data,
        rows,
        cols,
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=parameters,
        solver_options=options,
        name="K",
    )

    data = np.array(rhs["data"])
    rows = np.array(rhs["rows"])
    cols = np.array(rhs["cols"])
    indexptr = np.array(rhs["indexptr"])
    shape = (Ndofs, 1)
    rhs_op = COOMatrixOperator(
        (data, rows, cols),
        indexptr,
        dofmap.num_cells,
        shape,
        parameters=Parameters({}),
        solver_options=options,
        name="F",
    )
    return op, rhs_op, local_bases


class DirichletLift(object):
    def __init__(
        self,
        space: FenicsxVectorSpace,
        a_cpp: Union[df.fem.Form, list[Any], Any],
        facets: npt.NDArray[np.int32],
    ):
        self.range = space
        self._a = a_cpp
        self._x = df.la.create_petsc_vector(space.V.dofmap.index_map, space.V.dofmap.bs)  # type: ignore
        tdim = space.V.mesh.topology.dim  # type: ignore
        fdim = tdim - 1
        self._dofs = df.fem.locate_dofs_topological(space.V, fdim, facets)  # type: ignore
        self._g = df.fem.Function(space.V)  # type: ignore
        self._bcs = [df.fem.dirichletbc(self._g, self._dofs)]  # type: ignore
        self.dofs = self._bcs[0]._cpp_object.dof_indices()[0]

    def _update_dirichlet_data(self, values):
        self._g.x.petsc_vec.zeroEntries()  # type: ignore
        self._g.x.array[self.dofs] = values  # type: ignore
        self._g.x.scatter_forward()  # type: ignore

    def assemble(self, values):
        self._update_dirichlet_data(values)
        bcs = self._bcs
        self._x.zeroEntries()
        dolfinx.fem.petsc.apply_lifting(self._x, [self._a], bcs=[bcs])  # type: ignore
        self._x.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        dolfinx.fem.petsc.set_bc(self._x, bcs)

        return self.range.make_array([self._x])  # type: ignore


class ParametricTransferProblem(LogMixin):
    def __init__(
        self,
        operator: FenicsxMatrixBasedOperator,
        rhs: DirichletLift,
        range_space: FenicsxVectorSpace,
        source_product: Optional[Operator] = None,
        range_product: Optional[Operator] = None,
        kernel: Optional[VectorArray] = None,
        padding: float = 1e-14,
    ):
        assert rhs.range is operator.range
        self.operator = operator
        self.rhs = rhs
        self.source_product = source_product
        self.range_product = range_product
        self.kernel = kernel

        self.source = operator.source  # type: ignore
        self.range = range_space
        self._restriction = make_mapping(self.range.V, self.source.V, padding=padding, check=True)

    def generate_random_boundary_data(
        self, count: int, distribution: str, options: Optional[dict[str, Any]] = None
    ) -> npt.NDArray:
        """Generates random vectors of shape (count, num_dofs_Γ_out).

        Args:
            count: Number of random vectors.
            distribution: The distribution used for sampling.
            options: Arguments passed to sampling method of random number generator.

        """

        num_dofs = self.rhs.dofs.size
        options = options or {}
        values = create_random_values((count, num_dofs), distribution, **options)

        return values

    def assemble_operator(self, mu=None):
        self.logger.info(f"Assembling operator for {mu=}.")
        self.op = self.operator.assemble(mu) # type: ignore

    def solve(self, boundary_values) -> VectorArray:
        """Solve the transfer problem for given `boundary_values`.

        args:
            A: The assembled operator.
            boundary_values: Values to be prescribed on Gamma out.
        """
        assert isinstance(self.op, FenicsxMatrixOperator)

        # solution
        U_in = self.range.zeros(0, reserve=len(boundary_values))

        # construct rhs from boundary data
        for array in boundary_values:
            Ag = self.rhs.assemble(array)
            U = self.op.apply_inverse(Ag)

            # ### restrict full solution to target subdomain
            U_in.append(self.range.from_numpy(U.dofs(self._restriction)))

        if self.kernel is not None:
            assert len(self.kernel) > 0 # type: ignore
            return orthogonal_part(
                U_in, self.kernel, product=None, orthonormal=True
            )
        else:
            return U_in


def discretize_transfer_problem(example: BeamData, configuration: str) -> tuple[ParametricTransferProblem, VectorArray]:
    """Discretizes the transfer problem for given `configuration`.

    Args:
        example: Example data class.
        configuration: The configuration/archetype.
    """

    from .fom import ParaGeomLinEla
    from .auxiliary_problem import GlobalAuxiliaryProblem
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCGeom, BCTopo, _create_dirichlet_bcs
    from multi.preprocessing import create_meshtags

    global_coarse_domain, _, _ = read_mesh(example.coarse_grid("global"), MPI.COMM_WORLD,
                                           kwargs={"gdim": example.gdim})
    global_coarse_grid = StructuredQuadGrid(global_coarse_domain)

    # ### Structured coarse grid of the oversampling domain
    coarse_domain, _, _ = read_mesh(
        example.coarse_grid(configuration),
        MPI.COMM_WORLD,
        kwargs={"gdim": example.gdim},
    )
    coarse_grid = StructuredQuadGrid(coarse_domain)

    # locate interfaces for definition of auxiliary problem
    cell_vertices = coarse_grid.get_entities(0, 0)
    x_vertices = coarse_grid.get_entity_coordinates(0, cell_vertices)
    x_min = x_vertices[0][0]
    interface_locators = []
    for i in range(1, coarse_grid.num_cells):
        x_coord = float(x_min + i)
        interface_locators.append(plane_at(x_coord, "x"))

    # ### Fine scale grid of the oversampling domain Ω
    domain, ct, ft = read_mesh(
        example.parent_domain(configuration),
        MPI.COMM_WORLD,
        kwargs={"gdim": example.gdim},
    )
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)

    # ### Neumann data and Facet tag definition
    neumann_data = example.get_neumann(global_coarse_grid.grid, configuration)
    ft_def = {}
    for tag, key in zip([int(11), int(12), int(13)], ["bottom", "left", "right"]):
        ft_def[key] = (tag, omega.str_to_marker(key))

    top_tag = None
    top_locator = None
    if configuration == "left":
        assert neumann_data is not None
        top_tag, top_locator = neumann_data
        ft_def["top"] = (top_tag, top_locator)
    elif configuration == "inner":
        assert neumann_data is None
    elif configuration == "right":
        assert neumann_data is None
    else:
        raise NotImplementedError

    new_facet_tags, _ = create_meshtags(omega.grid, omega.tdim-1, ft_def, tags=omega.facet_tags)
    omega.facet_tags = new_facet_tags

    # Define tags for auxiliary problem and check facet tags
    aux_tags = None
    if configuration == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 4  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 0  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        assert omega.facet_tags.find(18).size == example.num_intervals * 4  # void 4
        aux_tags = [15, 16, 17, 18]

    elif configuration == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(top_tag).size == example.num_intervals * 1  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]

    elif configuration == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 0  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]
    else:
        raise NotImplementedError

    # ### Target subdomain Ω_in
    local_cell_index = example.config_to_omega_in(configuration, local=True)[0]
    assert local_cell_index in (0, 1)
    # get xmin to translate target subdomain by lower left corner point
    cell_vertices = coarse_grid.get_entities(0, local_cell_index)
    x_cell_verts = coarse_grid.get_entity_coordinates(0, cell_vertices)
    xmin_omega_in = x_cell_verts[0]
    target_domain, _, _ = read_mesh(
        example.target_subdomain, MPI.COMM_WORLD, kwargs={"gdim": example.gdim}
    )
    omega_in = RectangularDomain(target_domain)
    omega_in.translate(xmin_omega_in)

    # create necessary connectivities
    omega.grid.topology.create_connectivity(0, 2)
    omega_in.grid.topology.create_connectivity(0, 2)

    # ### Function Spaces
    V = df.fem.functionspace(omega.grid, ("P", example.geom_deg, (example.gdim,)))
    V_in = df.fem.functionspace(target_domain, V.ufl_element())
    target_space = FenicsxVectorSpace(V_in)

    # ### Dirichlet BCs
    # have to be defined twice (operator & range product)
    zero = df.fem.Constant(V.mesh, (df.default_scalar_type(0.0),) * example.gdim)
    global_cell_index = example.config_to_omega_in(configuration, local=False)[0]
    gamma_out = example.get_gamma_out(global_cell_index)
    bc_gamma_out = BCGeom(zero, gamma_out, V)
    bcs_op = list()
    bcs_op.append(bc_gamma_out)
    bcs_range_product = []
    hom_dirichlet = example.get_dirichlet(global_coarse_grid.grid, configuration)
    if hom_dirichlet is not None:
        for bc_spec in hom_dirichlet:
            # determine entities and define BCTopo
            entities_omega = df.mesh.locate_entities_boundary(
                V.mesh, bc_spec["entity_dim"], bc_spec["boundary"]
            )
            entities_omega_in = df.mesh.locate_entities_boundary(
                V_in.mesh, bc_spec["entity_dim"], bc_spec["boundary"]
            )
            bc = BCTopo(
                df.fem.Constant(V.mesh, bc_spec["value"]),
                entities_omega,
                bc_spec["entity_dim"],
                V,
                sub=bc_spec["sub"],
            )
            bc_rp = BCTopo(
                df.fem.Constant(V_in.mesh, bc_spec["value"]),
                entities_omega_in,
                bc_spec["entity_dim"],
                V_in,
                sub=bc_spec["sub"],
            )
            bcs_op.append(bc)
            bcs_range_product.append(bc_rp)
    bcs_op = tuple(bcs_op)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))
    assert len(bcs_op) - 1 == len(bcs_range_product)

    # ### Auxiliary Problem
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    problem = LinearElasticityProblem(omega, V, phases=mat)
    auxiliary_problem = GlobalAuxiliaryProblem(
        problem, aux_tags, example.parameters[configuration], coarse_grid, interface_locators=interface_locators
    )
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Discretize left hand side - FenicsxMatrixBasedOperator
    parageom = ParaGeomLinEla(
        omega,
        V,
        d=d_trafo,  # type: ignore
        matparam={"gdim": example.gdim, "E": 1.0, "NU": example.poisson_ratio, "plane_stress": example.plane_stress},
    )
    params = example.parameters[configuration]

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary_problem.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    operator = FenicsxMatrixBasedOperator(
        parageom.form_lhs, params, param_setter=param_setter, bcs=bcs_op
    )

    # ### Discretize right hand side - DirichletLift
    entities_gamma_out = df.mesh.locate_entities_boundary(
        V.mesh, V.mesh.topology.dim - 1, gamma_out
    )
    assert entities_gamma_out.size > 0
    rhs = DirichletLift(operator.range, operator.compiled_form, entities_gamma_out)  # type: ignore

    def h1_0_semi(V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx # type: ignore

    def l2(V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return ufl.inner(u, v) * ufl.dx # type: ignore

    # ### Range product operator
    h1_cpp = df.fem.form(h1_0_semi(V_in))
    pmat_range = dolfinx.fem.petsc.create_matrix(h1_cpp)
    pmat_range.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_range, h1_cpp, bcs=bcs_range_product)
    pmat_range.assemble()
    range_product = FenicsxMatrixOperator(pmat_range, V_in, V_in, name="h1_0_semi")

    # ### Source product operator
    l2_cpp = df.fem.form(l2(V))
    pmat_source = dolfinx.fem.petsc.create_matrix(l2_cpp)
    pmat_source.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_source, l2_cpp, bcs=[])
    pmat_source.assemble()
    source_mat = csr_array(pmat_source.getValuesCSR()[::-1])  # type: ignore
    source_product = NumpyMatrixOperator(source_mat[rhs.dofs, :][:, rhs.dofs], name="l2")

    # ### Rigid body modes
    kernel_set = example.get_kernel_set(global_cell_index)
    ns_vecs = build_nullspace(V_in, gdim=example.gdim)
    assert len(ns_vecs) == 3
    rigid_body_modes = []
    for j in kernel_set:
        dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
        rigid_body_modes.append(ns_vecs[j])
    kernel = target_space.make_array(rigid_body_modes)  # type: ignore
    gram_schmidt(kernel, product=None, copy=False)
    assert np.allclose(kernel.gramian(), np.eye(len(kernel)))

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        target_space,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
        padding=1e-8,
    )

    # ### Discretize Neumann Data
    if configuration == "left":
        dA = ufl.Measure("ds", domain=omega.grid, subdomain_data=omega.facet_tags)
        t_y = -example.traction_y / example.youngs_modulus
        traction = df.fem.Constant(
            omega.grid,
            (df.default_scalar_type(0.0), df.default_scalar_type(t_y)),
        )
        v = ufl.TestFunction(V)
        L = ufl.inner(v, traction) * dA(ft_def["top"][0])
        Lcpp = df.fem.form(L)
        f_ext = dolfinx.fem.petsc.create_vector(Lcpp)  # type: ignore

        with f_ext.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(f_ext, Lcpp)

        # Apply boundary conditions to the rhs
        bcs_neumann = _create_dirichlet_bcs(bcs_op)
        dolfinx.fem.petsc.apply_lifting(f_ext, [operator.compiled_form], bcs=[bcs_neumann])  # type: ignore
        f_ext.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        dolfinx.fem.petsc.set_bc(f_ext, bcs_neumann)

        assert np.isclose(np.sum(f_ext.array), -example.traction_y / example.youngs_modulus)
        F_ext = operator.range.make_array([f_ext])  # type: ignore
    else:
        F_ext = operator.range.zeros(1)

    return transfer, F_ext


if __name__ == "__main__":
    from .tasks import example
    # discretize_transfer_problem(example, "left")
    discretize_transfer_problem(example, "right")
    # discretize_transfer_problem(example, "inner")
