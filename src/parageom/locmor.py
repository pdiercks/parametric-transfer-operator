from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import dolfinx as df
import dolfinx.fem.petsc
from mpi4py import MPI
from multi.boundary import plane_at, point_at, within_range
from multi.dofmap import DofMap
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.interpolation import make_mapping
from multi.io import read_mesh
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.projection import orthogonal_part
from multi.sampling import create_random_values
from multi.solver import build_nullspace
from multi.utils import LogMixin
import numpy as np
import numpy.typing as npt
from petsc4py import PETSc
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.operators.constructions import LincombOperator, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace
import scipy.linalg
from scipy.sparse import coo_array, csr_array
import ufl

from parageom.definitions import BeamData
from parageom.dofmap_gfem import GFEMDofMap
from parageom.matrix_based_operator import FenicsxMatrixBasedOperator


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

    def assemble(self, mu=None): # type: ignore
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

    def assemble(self, mu=None): # type: ignore
        assert self.parameters.assert_compatible(mu)
        assert self.parametric
        assert mu is not None

        op = self.ei_sub_op.rop # type: ignore
        magic_dofs = self.ei_sub_op.magic_dofs # type: ignore
        m_inv = self.ei_sub_op.m_inv # type: ignore
        rdofs = op.restricted_range_dofs[m_inv].reshape(magic_dofs.shape)
        interpolation_matrix = self.ei_sub_op.interpolation_matrix # type: ignore
        indexptr = self.indexptr # type: ignore
        M = len(self.ei_sub_op.cb) # type: ignore

        data = self.data # type: ignore
        new = np.zeros((data.shape[1],), dtype=np.float64)

        for i, mu_i in enumerate(mu.to_numpy()):
            # restricted evaluation of the subdomain operator
            loc_mu = op.parameters.parse([mu_i])
            A = csr_array(op.assemble(loc_mu).matrix.getValuesCSR()[::-1])
            _coeffs = scipy.linalg.solve(interpolation_matrix, A[rdofs[:, 0], rdofs[:, 1]])
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
    u_global_view[:] = 0.0

    for cell in range(dofmap.num_cells):
        # translate subdomain mesh
        vertices = coarse_grid.get_entities(0, cell)
        dx_cell = coarse_grid.get_entity_coordinates(0, vertices)[0]
        x_submesh += dx_cell

        # fill u_local with rom solution
        basis = bases[cell]
        dofs = dofmap.cell_dofs(cell)

        # fill global field via dof mapping
        V_to_Vsub = make_mapping(Vsub, V, padding=1e-8, check=True)
        u_global_view[V_to_Vsub] = U_rb[0, dofs] @ basis

        # move subdomain mesh to origin
        x_submesh -= dx_cell
    u_global.x.scatter_forward()


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

    from parageom.dofmap_gfem import select_modes

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
    data = np.array(lhs["data"], dtype=np.float64)
    rows = np.array(lhs["rows"], dtype=np.int32)
    cols = np.array(lhs["cols"], dtype=np.int32)
    indexptr = np.array(lhs["indexptr"], dtype=np.int32)
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

    data = np.array(rhs["data"], dtype=np.float64)
    rows = np.array(rhs["rows"], dtype=np.int32)
    cols = np.array(rhs["cols"], dtype=np.int32)
    indexptr = np.array(rhs["indexptr"], dtype=np.int32)
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

    from parageom.dofmap_gfem import select_modes

    # TODO: try to set BCs to better condition system matrix

    # no need to apply bcs as basis functions should
    # satisfy these automatically
    # raise NotImplementedError("FIXME, condition number")
    # left = dofmap.grid.locate_entities_boundary(0, plane_at(0.0, "x"))
    # right = dofmap.grid.locate_entities_boundary(0, point_at([10., 0., 0.]))

    # assert left.size == 2
    # assert right.size == 1

    # bc_dofs = []
    # for vertex in left:
    #     bc_dofs.append(dofmap.entity_dofs(vertex)[0])
    # for vertex in right:
    #     bc_dofs.append(dofmap.entity_dofs(vertex)[0])
    #     bc_dofs.append(dofmap.entity_dofs(vertex)[1])
    # bc_dofs = np.array(bc_dofs, dtype=np.int32)
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
    data = np.vstack(_data, dtype=np.float64)
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

    # ### LHS
    # Matrix operator
    rows = np.array(lhs["rows"], dtype=np.int32)
    cols = np.array(lhs["cols"], dtype=np.int32)
    indexptr = np.array(lhs["indexptr"], dtype=np.int32)
    shape = (Ndofs, Ndofs)
    options = None
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
    # BC operator
    if np.any(bc_dofs):
        bc_array = coo_array(
            (bc_mat["data"], (bc_mat["rows"], bc_mat["cols"])), shape=shape
        )
        bc_array.eliminate_zeros()
        bc_op = NumpyMatrixOperator(
            bc_array.tocsr(), op.source.id, op.range.id, op.solver_options, "bc_mat"
        )
        lhs_op = LincombOperator([op, bc_op], [1.0, 1.0])
    else:
        lhs_op = op

    # ### RHS
    data = np.array(rhs["data"], dtype=np.float64)
    rows = np.array(rhs["rows"], dtype=np.int32)
    cols = np.array(rhs["cols"], dtype=np.int32)
    indexptr = np.array(rhs["indexptr"], dtype=np.int32)
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
    return lhs_op, rhs_op, local_bases


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
        self._restriction = make_mapping(self.range.V, self.source.V, padding=padding, check=True) # type: ignore

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
            U_in.append(self.range.from_numpy(U.dofs(self._restriction))) # type: ignore

        if self.kernel is not None:
            assert len(self.kernel) > 0 # type: ignore
            return orthogonal_part(
                U_in, self.kernel, product=None, orthonormal=True
            )
        else:
            return U_in


@dataclass
class OversamplingConfig:
    index: int
    cells_omega: npt.NDArray[np.int32]
    cells_omega_in: npt.NDArray[np.int32]
    kernel: tuple[int]
    gamma_out: Callable
    gamma_d: Optional[Callable] = None
    gamma_n: Optional[Callable] = None


def oversampling_config_factory(k):
    """Creates instance of `OversamplingConfig`"""

    cells_omega = {
            0: np.array([0, 1], dtype=np.int32),
            1: np.array([0, 1, 2], dtype=np.int32),
            2: np.array([0, 1, 2, 3], dtype=np.int32),
            3: np.array([1, 2, 3, 4], dtype=np.int32),
            4: np.array([2, 3, 4, 5], dtype=np.int32),
            5: np.array([3, 4, 5, 6], dtype=np.int32),
            6: np.array([4, 5, 6, 7], dtype=np.int32),
            7: np.array([5, 6, 7, 8], dtype=np.int32),
            8: np.array([6, 7, 8, 9], dtype=np.int32),
            9: np.array([7, 8, 9], dtype=np.int32),
            10: np.array([8, 9], dtype=np.int32),
            }

    cells_omega_in = {
            0: np.array([0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([1, 2], dtype=np.int32),
            3: np.array([2, 3], dtype=np.int32),
            4: np.array([3, 4], dtype=np.int32),
            5: np.array([4, 5], dtype=np.int32),
            6: np.array([5, 6], dtype=np.int32),
            7: np.array([6, 7], dtype=np.int32),
            8: np.array([7, 8], dtype=np.int32),
            9: np.array([8, 9], dtype=np.int32),
            10: np.array([9], dtype=np.int32),
            }

    kernel = {
            0: (1, 2),
            1: (1, 2),
            2: (0, 1, 2),
            3: (0, 1, 2),
            4: (0, 1, 2),
            5: (0, 1, 2),
            6: (0, 1, 2),
            7: (0, 1, 2),
            8: (0, 1, 2),
            9: (0, 2),
            10: (0, 2),
            }
    # required enrichment should be determined from kernel
    # see src/parageom/gfem.py

    # ### Topology
    x_max = 10.0
    y_max = 1.0
    unit_length = 1.0
    support_left = plane_at(0.0, "x")
    support_right = point_at([x_max, 0.0, 0.0])
    neumann_top = within_range([0.0, y_max, 0.0], [unit_length, y_max, 0.0])

    # define left based on smallest cell integer
    left_most_cell = np.amin(cells_omega[k])
    x_left = float(left_most_cell * unit_length)
    right_most_cell = np.amax(cells_omega[k])
    x_right = float((right_most_cell + 1) * unit_length)
    # define right based on largest cell integer
    tol = 1e-4
    left = within_range([x_left, 0.0 + tol, 0.0], [x_left, y_max - tol, 0.0])
    right = within_range([x_right, 0.0 + tol, 0.0], [x_right, y_max - tol, 0.0])
    def gamma_out_inner(x):
        return np.logical_or(left(x), right(x))
    gamma_out = {
            0: right,
            1: right,
            2: right,
            3: gamma_out_inner,
            4: gamma_out_inner,
            5: gamma_out_inner,
            6: gamma_out_inner,
            7: gamma_out_inner,
            8: left,
            9: left,
            10: left,
            }
    if k in (0, 1, 2):
        return OversamplingConfig(k, cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k], gamma_d=support_left, gamma_n=neumann_top)
    elif k in (8, 9, 10):
        return OversamplingConfig(k, cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k], gamma_d=support_right, gamma_n=None)
    else:
        return OversamplingConfig(k, cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k])


def discretize_transfer_problem(example: BeamData, struct_grid_gl: StructuredQuadGrid, osp_config: OversamplingConfig, debug: bool=False):
    """Discretize transfer problem.

    Args:
        example: The data class for the example problem.
        struct_grid_gl: Global coarse grid.
        osp_config: Configuration of this transfer problem.
    """
    from parageom.preprocessing import create_structured_coarse_grid_v2, create_fine_scale_grid_v2
    from parageom.auxiliary_problem import GlobalAuxiliaryProblem
    from parageom.fom import ParaGeomLinEla
    from parageom.matrix_based_operator import _create_dirichlet_bcs, BCTopo, BCGeom
    from parageom.locmor import ParametricTransferProblem, DirichletLift
    from multi.preprocessing import create_meshtags

    cells_omega = osp_config.cells_omega
    cells_omega_in = osp_config.cells_omega_in

    # create coarse grid partition of oversampling domain
    outstream = example.path_omega_coarse(osp_config.index)
    create_structured_coarse_grid_v2(example, struct_grid_gl, cells_omega, outstream.as_posix())
    coarse_omega = read_mesh(outstream, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})[0]
    struct_grid_omega = StructuredQuadGrid(coarse_omega)
    assert struct_grid_omega.num_cells == cells_omega.size

    # create fine grid partition of oversampling domain
    output = example.path_omega(osp_config.index)
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega, output.as_posix())
    omega, omega_ct, omega_ft = read_mesh(output, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})
    omega = RectangularDomain(omega, cell_tags=omega_ct, facet_tags=omega_ft)
    # create facets
    # facet tags for void interfaces start from 15 (see create_fine_scale_grid_v2)
    # i.e. 15 ... 24 for max number of cells

    facet_tag_definitions = {}
    for tag, key in zip([int(11), int(12), int(13)], ["bottom", "left", "right"]):
        facet_tag_definitions[key] = (tag, omega.str_to_marker(key))

    # add tags for neumann boundary
    top_tag = None
    if osp_config.gamma_n is not None:
        top_tag = int(194)
        top_locator = osp_config.gamma_n
        facet_tag_definitions["top"] = (top_tag, top_locator)

    # update already existing facet tags
    # this will add tags for "top" boundary
    omega.facet_tags = create_meshtags(omega.grid, omega.tdim-1, facet_tag_definitions, tags=omega.facet_tags)[0]

    assert omega.facet_tags.find(11).size == example.num_intervals * cells_omega.size  # bottom
    assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
    assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
    for itag in range(15, 15 + cells_omega.size):
        assert omega.facet_tags.find(itag).size == example.num_intervals * 4  # void
    # create fine grid partition of target subdomain
    output = example.path_omega_in(osp_config.index)
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega_in, output.as_posix())
    omega_in, omega_in_ct, omega_in_ft = read_mesh(output, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})
    omega_in = RectangularDomain(omega_in, cell_tags=omega_in_ct, facet_tags=omega_in_ft)

    # create necessary connectivities
    omega.grid.topology.create_connectivity(0, 2)
    omega_in.grid.topology.create_connectivity(0, 2)

    # ### Function Spaces
    V = df.fem.functionspace(omega.grid, ("P", example.geom_deg, (example.gdim,)))
    V_in = df.fem.functionspace(omega_in.grid, V.ufl_element())
    source_V_in = FenicsxVectorSpace(V_in)

    # ### Auxiliary problem defined on oversampling domain Omega
    # locate interfaces for definition of auxiliary problem
    left_most_cell = np.amin(cells_omega)
    unit_length = 1.0
    x_min = float(left_most_cell * unit_length)

    interface_locators = []
    for i in range(1, cells_omega.size):
        x_coord = float(x_min + i)
        interface_locators.append(plane_at(x_coord, "x"))

    if debug:
        for marker in interface_locators:
            entities = df.mesh.locate_entities(V.mesh, V.mesh.topology.dim-1, marker)
            assert entities.size == example.num_intervals

    aux_tags = list(range(15, 15 + cells_omega.size))
    assert len(aux_tags) == cells_omega.size
    assert len(interface_locators) == cells_omega.size - 1
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    problem = LinearElasticityProblem(omega, V, phases=mat)
    params = Parameters({"R": cells_omega.size})
    auxiliary_problem = GlobalAuxiliaryProblem(
        problem, aux_tags, params, struct_grid_omega, interface_locators=interface_locators
    )
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Dirichlet BCs (operator, range product)
    bcs_op = [] # BCs for lhs operator of transfer problem, space V
    bcs_range_product = [] # BCs for range product operator, space V_in

    zero = df.default_scalar_type(0.0)
    fix_u = df.fem.Constant(V.mesh, (zero,) * example.gdim)
    bc_gamma_out = BCGeom(fix_u, osp_config.gamma_out, V)
    bcs_op.append(bc_gamma_out)

    dirichlet_bc = []
    if osp_config.index in (0, 1, 2):
        # left Dirichlet boundary is active
        dirichlet_bc.append({
            "value": zero,
            "boundary": osp_config.gamma_d,
            "entity_dim": 1,
            "sub": 0})
    elif osp_config.index in (8, 9, 10):
        # right Dirichlet boundary is active
        dirichlet_bc.append({
            "value": zero,
            "boundary": osp_config.gamma_d,
            "entity_dim": 0,
            "sub": 1})

    for bc_spec in dirichlet_bc:
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

    # ### Discretize left hand side - FenicsxMatrixBasedOperator
    matparam = {"gdim": example.gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(
        omega,
        V,
        d=d_trafo,  # type: ignore
        matparam=matparam,
    )

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary_problem.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    # operator for left hand side on full oversampling domain
    operator = FenicsxMatrixBasedOperator(
        parageom.form_lhs, params, param_setter=param_setter, bcs=bcs_op
    )

    # ### Discretize right hand side - DirichletLift
    entities_gamma_out = df.mesh.locate_entities_boundary(
        V.mesh, V.mesh.topology.dim - 1, osp_config.gamma_out
    )
    expected_num_facets_gamma_out = (example.num_intervals - 2, 2 * (example.num_intervals - 2))
    assert entities_gamma_out.size in expected_num_facets_gamma_out
    rhs = DirichletLift(operator.range, operator.compiled_form, entities_gamma_out)  # type: ignore


    def l2(V):
        """form for source product"""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return ufl.inner(u, v) * ufl.dx # type: ignore

    # ### Source product operator
    l2_cpp = df.fem.form(l2(V))
    pmat_source = dolfinx.fem.petsc.create_matrix(l2_cpp) # type: ignore
    pmat_source.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_source, l2_cpp, bcs=[])
    pmat_source.assemble()
    source_mat = csr_array(pmat_source.getValuesCSR()[::-1])  # type: ignore
    source_product = NumpyMatrixOperator(source_mat[rhs.dofs, :][:, rhs.dofs], name="l2")

    # ### Range Product
    range_mat = LinearElasticMaterial(**matparam)
    linela_target = LinearElasticityProblem(omega_in, V_in, phases=range_mat)
    a_cpp = df.fem.form(linela_target.form_lhs)
    range_mat = dolfinx.fem.petsc.create_matrix(a_cpp) # type: ignore
    range_mat.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(range_mat, a_cpp, bcs=bcs_range_product)
    range_mat.assemble()
    range_product = FenicsxMatrixOperator(range_mat, V_in, V_in, name="energy")

    # ### Rigid body modes
    kernel_set = osp_config.kernel
    ns_vecs = build_nullspace(V_in, gdim=example.gdim)
    assert len(ns_vecs) == 3
    rigid_body_modes = []

    kernel = None
    if len(kernel_set) > 0:
        for j in kernel_set:
            dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
            rigid_body_modes.append(ns_vecs[j])
        kernel = source_V_in.make_array(rigid_body_modes)  # type: ignore
        gram_schmidt(kernel, product=None, copy=False)
        assert np.allclose(kernel.gramian(), np.eye(len(kernel)))
    assert kernel is not None

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        source_V_in,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
        padding=1e-8,
    )

    if osp_config.gamma_n is not None:
        assert top_tag is not None
        assert omega.facet_tags.find(top_tag).size == example.num_intervals * 1  # top
        dA = ufl.Measure("ds", domain=omega.grid, subdomain_data=omega.facet_tags)
        t_y = -example.traction_y
        traction = df.fem.Constant(
            omega.grid,
            (df.default_scalar_type(0.0), df.default_scalar_type(t_y)),
        )
        v = ufl.TestFunction(V)
        L = ufl.inner(v, traction) * dA(facet_tag_definitions["top"][0])
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

        assert np.isclose(np.sum(f_ext.array), -example.traction_y)
        F_ext = operator.range.make_array([f_ext])  # type: ignore
    else:
        F_ext = operator.range.zeros(1)


    return transfer, F_ext


if __name__ == "__main__":
    print("hi from locmor")
    # TODO add basic test?
