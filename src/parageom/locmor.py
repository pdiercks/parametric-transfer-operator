from typing import Tuple, Optional

from collections import defaultdict, namedtuple

import numpy as np
from scipy.sparse import coo_array, csr_array
from scipy.linalg import solve

from mpi4py import MPI
import dolfinx as df
from dolfinx.fem.petsc import set_bc
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.projection import project
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.operators.constructions import VectorOperator, LincombOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Parameters, ParameterSpace

from multi.boundary import point_at
from multi.domain import RectangularDomain, RectangularSubdomain
from multi.dofmap import DofMap
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinElaSubProblem, TransferProblem
from multi.product import InnerProduct
from multi.solver import build_nullspace
from multi.io import read_mesh, select_modes
from multi.interpolation import make_mapping
from .definitions import BeamData, BeamProblem
from .matrix_based_operator import FenicsxMatrixBasedOperator


EISubdomainOperatorWrapper = namedtuple("EISubdomainOperator", ["rop", "cb", "interpolation_matrix"])


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

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray], indexptr: np.ndarray, num_cells: int, shape: Tuple[int, int], parameters: Parameters = {}, solver_options: Optional[dict] = None, name: Optional[str] = None):
        assert all([d.shape == data[0].shape for d in data])
        self.__auto_init(locals()) # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])
        self._data = data[0].copy()

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)

        data, rows, cols = self.data # type: ignore
        indexptr = self.indexptr # type: ignore
        num_cells = self.num_cells # type: ignore

        new = self._data
        if self.parametric and mu is not None:
            m = mu.to_numpy()
            new[:indexptr[0]] = data[:indexptr[0]] * m[0]
            for i in range(1, num_cells):
                new[indexptr[i-1]:indexptr[i]] = data[indexptr[i-1]:indexptr[i]] * m[i]

        K = coo_array((new, (rows, cols)), shape=self.shape) # type: ignore
        K.eliminate_zeros()
        return NumpyMatrixOperator(K.tocsr(), self.source.id, self.range.id, self.solver_options, self.name + "_assembled")

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)


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
        rdofs = op.restricted_range_dofs
        interpolation_matrix = self.ei_sub_op.interpolation_matrix
        indexptr = self.indexptr
        M = len(self.ei_sub_op.cb)

        data = self.data
        new = np.zeros((data.shape[1],), dtype=np.float32)

        # TODO: works as expected?

        for i, mu_i in enumerate(mu.to_numpy()):
            # restricted evaluation of the subdomain operator
            loc_mu = op.parameters.parse([mu_i])
            A = csr_array(op.assemble(loc_mu).matrix.getValuesCSR()[::-1])
            _coeffs = solve(interpolation_matrix, A[rdofs, rdofs])
            λ = _coeffs.reshape(1, M)

            subdomain_range = np.s_[indexptr[i-1]:indexptr[i]]
            if i == 0:
                subdomain_range = np.s_[0:indexptr[i]]
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

        # move subdomain mesh to origin
        x_submesh -= dx_cell
    u_global.x.scatter_forward()


def assemble_system(
    example,
    num_modes: int,
    dofmap: DofMap,
    operator: FenicsxMatrixBasedOperator,
    b: VectorOperator,
    mu,
    bases: list[np.ndarray],
    num_max_modes: np.ndarray,
    parameters: Parameters
):
    """Assembles ``operator`` and ``rhs`` for localized ROM as ``StationaryModel``.

    Args:
        example: The example data class.
        num_modes: Number of fine scale modes per edge to be used.
        dofmap: The dofmap of the global reduced space.
        operator: Local high fidelity operator for stiffness matrix.
        b: Local high fidelity external force vector.
        mu: Parameter value.
        bases: Local reduced basis for each subdomain.
        num_max_modes: Maximum number of fine scale modes for each edge.
        parameters: The |Parameters| the ROM depends on.

    """
    dofs_per_vertex = 2
    dofs_per_face = 0

    dofs_per_edge = num_max_modes.copy()
    dofs_per_edge[num_max_modes > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vertex, dofs_per_edge, dofs_per_face)

    # ### Definition of Dirichlet BCs
    # This also depends on number of modes and can only be defined after
    # distribution of dofs
    origin = dofmap.grid.locate_entities_boundary(0, point_at([0.0, 0.0, 0.0]))
    bottom_right = dofmap.grid.locate_entities_boundary(0, point_at([example.length, 0.0, 0.0]))
    bc_dofs = []
    for vertex in origin:
        bc_dofs += dofmap.entity_dofs(0, vertex)
    for vertex in bottom_right:
        dofs = dofmap.entity_dofs(0, vertex)
        bc_dofs.append(dofs[1])  # constrain uy, but not ux
    assert len(bc_dofs) == 3
    bc_dofs = np.array(bc_dofs)

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
        local_basis = select_modes(bases[ci], num_max_modes[ci], dofs_per_edge[ci])
        local_bases.append(local_basis)
        B = A.source.from_numpy(local_basis)  # type: ignore
        A_local = project(A, B, B)
        b_local = project(b, B, None)
        element_matrix = A_local.matrix  # type: ignore
        element_vector = b_local.matrix  # type: ignore
        print(f"{ci=}")
        print(f"{element_matrix[0, 0]=}")
        # print(f"{element_vector=}")

        for l, x in enumerate(dofs):
            if x in bc_dofs:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(0.0)
            else:
                rhs["rows"].append(x)
                rhs["cols"].append(0)
                rhs["data"].append(element_vector[l, 0])

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
                    lhs["data"].append(element_matrix[l, k])

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

    # ### Add matrix to account for BCs
    bc_array = coo_array(
        (bc_mat["data"], (bc_mat["rows"], bc_mat["cols"])), shape=shape
    )
    bc_array.eliminate_zeros()
    bc_op = NumpyMatrixOperator(
        bc_array.tocsr(), op.source.id, op.range.id, op.solver_options, "bc_mat"
    )

    lincomb = LincombOperator([op, bc_op], [1.0, 1.0])

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
    return lincomb, rhs_op, local_bases


def assemble_system_with_ei(
    example,
    num_modes: int,
    dofmap: DofMap,
    ei_sub_op: EISubdomainOperatorWrapper,
    b: VectorOperator,
    bases: list[np.ndarray],
    num_max_modes: np.ndarray,
    parameters: Parameters
):
    """Assembles ``operator`` and ``rhs`` for localized ROM as ``StationaryModel``.

    Args:
        num_modes: Number of fine scale modes per edge to be used.
        dofmap: The dofmap of the global reduced space.
        mops: Collateral basis (matrix operators) of subdomain operator.
        b: Local high fidelity external force vector.
        bases: Local reduced basis for each subdomain.
        num_max_modes: Maximum number of fine scale modes for each edge.
        parameters: The |Parameters| the ROM depends on.

    """

    dofs_per_vertex = 2
    dofs_per_face = 0

    dofs_per_edge = num_max_modes.copy()
    dofs_per_edge[num_max_modes > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vertex, dofs_per_edge, dofs_per_face)

    # ### Definition of Dirichlet BCs
    # This also depends on number of modes and can only be defined after
    # distribution of dofs
    length = example.length
    origin = dofmap.grid.locate_entities_boundary(0, point_at([0.0, 0.0, 0.0]))
    bottom_right = dofmap.grid.locate_entities_boundary(0, point_at([length, 0.0, 0.0]))
    bc_dofs = []
    for vertex in origin:
        bc_dofs += dofmap.entity_dofs(0, vertex)
    for vertex in bottom_right:
        dofs = dofmap.entity_dofs(0, vertex)
        bc_dofs.append(dofs[1])  # constrain uy, but not ux
    assert len(bc_dofs) == 3
    bc_dofs = np.array(bc_dofs)

    lhs = defaultdict(list)
    rhs = defaultdict(list)
    bc_mat = defaultdict(list)
    local_bases = []

    mops = ei_sub_op.cb
    source = mops[0].source
    M = len(mops)  # size of collateral basis

    for ci in range(dofmap.num_cells):
        dofs = dofmap.cell_dofs(ci)

        # select active modes
        local_basis = select_modes(bases[ci], num_max_modes[ci], dofs_per_edge[ci])
        local_bases.append(local_basis)
        B = source.from_numpy(local_basis)  # type: ignore
        mops_local = [project(A, B, B) for A in mops]
        b_local = project(b, B, None)
        element_matrices = [a_local.matrix for a_local in mops_local]  # type: ignore
        element_vector = b_local.matrix  # type: ignore

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
                            for m in range(M):
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
    for m in range(M):
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

    # ### Add matrix to account for BCs
    bc_array = coo_array(
        (bc_mat["data"], (bc_mat["rows"], bc_mat["cols"])), shape=shape
    )
    bc_array.eliminate_zeros()
    bc_op = NumpyMatrixOperator(
        bc_array.tocsr(), op.source.id, op.range.id, op.solver_options, "bc_mat"
    )

    lincomb = LincombOperator([op, bc_op], [1.0, 1.0])

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
    return lincomb, rhs_op, local_bases


def discretize_oversampling_problem(example: BeamData, configuration: str, index: int):
    """Returns TransferProblem for fixed parameter Mu.

    Args:
        example: The instance of the example dataclass.
        configuration: The type of oversampling problem.
        index: The index of the parameter value of the training set.

    """

    # use MPI.COMM_SELF for embarrassingly parallel workloads
    oversamplingdomain_xdmf = example.oversampling_domain(
        configuration, index
    ).as_posix()
    with XDMFFile(MPI.COMM_SELF, oversamplingdomain_xdmf, "r") as fh:
        domain = fh.read_mesh(name="Grid")

    omega = RectangularDomain(domain, cell_tags=None, facet_tags=None)
    omega.create_facet_tags(
        {"bottom": int(11), "left": int(12), "right": int(13), "top": int(14)}
    )

    if configuration == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3  # top

    elif configuration == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top

    elif configuration == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top

    else:
        raise NotImplementedError

    # ### Definitions dependent on configuration
    # Topology: Γ_out, Ω_in, Σ_D
    # Dirichlet BCs on Σ_D

    # BeamProblem is only used to get stuff for transfer problem definition
    # here we do not need the actual physical meshes?
    beamproblem = BeamProblem(
        example.coarse_grid("global"), example.global_parent_domain, example
    )
    cell_index = beamproblem.config_to_cell(configuration)
    gamma_out = beamproblem.get_gamma_out(cell_index)
    dirichlet = beamproblem.get_dirichlet(cell_index)
    kernel_set = beamproblem.get_kernel_set(cell_index)

    # ### Omega in
    gdim = example.gdim
    target_subdomain, _, _ = read_mesh(
        example.target_subdomain(configuration, index), MPI.COMM_WORLD, gdim=gdim
    )
    id_omega_in = 99
    omega_in = RectangularSubdomain(id_omega_in, target_subdomain)
    # create coarse grid of target subdomain
    # required for fine scale part computation using coarse FE space
    omega_in.create_coarse_grid(1)
    omega_in.create_boundary_grids()

    # ### FE spaces
    degree = example.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = df.fem.functionspace(omega.grid, fe)  # full space
    W = df.fem.functionspace(omega_in.grid, fe)  # range space

    # ### Oversampling problem
    emod = example.youngs_modulus
    nu = example.poisson_ratio
    material = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=False)
    oversampling_problem = LinearElasticityProblem(omega, V, phases=material)

    # ### Problem on target subdomain
    # definition of correct material for consistency
    # however, unless energy inner product is used as inner product for the range
    # space, this should not have influence on the solution
    subproblem = LinElaSubProblem(omega_in, W, phases=material)
    # required for fine scale part computation using coarse FE space
    subproblem.setup_coarse_space()
    subproblem.setup_edge_spaces()
    subproblem.create_map_from_V_to_L()

    # ### Range product operator
    # get homogeneous Dirichlet bcs if present
    bc_hom = []
    if dirichlet is not None:
        subproblem.add_dirichlet_bc(**dirichlet)
        bc_hom = subproblem.get_dirichlet_bcs()

    inner_product = InnerProduct(subproblem.V, example.range_product, bcs=bc_hom)
    pmat = inner_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(pmat, subproblem.V, subproblem.V)

    # ### Rigid body modes
    ns_vecs = build_nullspace(subproblem.V, gdim=omega_in.grid.geometry.dim)
    range_space = FenicsxVectorSpace(subproblem.V)
    rigid_body_modes = []
    for j in kernel_set:
        set_bc(ns_vecs[j], bc_hom)
        rigid_body_modes.append(ns_vecs[j])
    kernel = range_space.make_array(rigid_body_modes)
    gram_schmidt(kernel, product=range_product, copy=False)

    # ### TransferProblem
    oversampling_problem.clear_bcs()
    subproblem.clear_bcs()
    transfer = TransferProblem(
        oversampling_problem,
        subproblem,
        gamma_out,
        dirichlet=dirichlet,
        source_product={"product": "l2", "bcs": ()},
        range_product=range_product,
        kernel=kernel,
    )
    return transfer


if __name__ == "__main__":
    from .tasks import example
    from multi.misc import x_dofs_vectorspace, locate_dofs
    from pymor.bindings.fenicsx import FenicsxVisualizer

    param = example.parameters["right"]
    ps = ParameterSpace(param, example.mu_range)
    mu = ps.parameters.parse([0.15 * example.unit_length for _ in range(2)])
    configuration = "right"
    # configuration = "left"
    index = 0
    T = discretize_oversampling_problem(example, configuration, index)
    v = T.generate_random_boundary_data(1, distribution="normal")
    v[:, ::2] = 0.1  # set x component to value
    v[:, 1::2] = 0.1  # set y component to value
    U = T.solve(v)

    xdofs = x_dofs_vectorspace(T.range.V)
    if configuration == "left":
        dofs = locate_dofs(xdofs, np.array([[0.0, 0.0, 0.0]]))
    elif configuration == "right":
        dofs = locate_dofs(xdofs, np.array([[example.length, 0.0, 0.0]]))
    assert np.allclose(U.dofs(dofs)[:, 1], np.zeros_like(U.dofs(dofs)[:, 1]))

    viz = FenicsxVisualizer(T.range)
    viz.visualize(U, filename="./homDirichlet.bp")
