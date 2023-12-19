from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_array

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io.utils import XDMFFile
from basix.ufl import element
import ufl

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Parameters, ParameterSpace
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.operators.constructions import LincombOperator

from multi.boundary import plane_at, within_range
from multi.domain import Domain
from multi.product import InnerProduct
from multi.solver import build_nullspace
from .definitions import Example


class COOMatrixOperator(Operator):
    """Wraps COO matrix data as an |Operator|.

    Args:
        data: COO matrix data. See scipy.sparse.coo_array.
        indexptr: Points to end of data for each cell.
        num_cells: Number of cells.
        shape: The shape of the matrix.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray], indexptr: np.ndarray, num_cells: int, shape: Tuple[int, int], solver_options: Optional[dict] = None, name: Optional[str] = None):
        assert all([d.shape == data[0].shape for d in data])
        self.__auto_init(locals()) # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        data, rows, cols = self.data # type: ignore
        indexptr = self.indexptr # type: ignore
        num_cells = self.num_cells # type: ignore

        if mu is not None:
            m = mu.to_numpy()
            data[:indexptr[0]] *= m[0]
            for i in range(1, num_cells):
                data[indexptr[i-1]:indexptr[i]] *= m[i]

        K = coo_array((data, (rows, cols)), shape=self.shape) # type: ignore
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


def discretize_oversampling_problem(example: Example):
    """provides data to build T: LincombOperator A, ParameterSpace, dirichlet_dofs, range_dofs, projection_matrix"""

    with XDMFFile(MPI.COMM_WORLD, example.fine_oversampling_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, "subdomains")

    omega = Domain(domain)
    tdim = domain.topology.dim
    # fdim = tdim - 1
    gdim = domain.ufl_cell().geometric_dimension()

    # ### Gamma out
    left = plane_at(omega.xmin[0], "x")
    right = plane_at(omega.xmax[0], "x")
    gamma_out = lambda x: np.logical_or(left(x), right(x))
    # facets_gamma_out = mesh.locate_entities_boundary(domain, fdim, gamma_out)

    # ### Omega in
    mark_omega_in = within_range([1., 0.], [2., 1.])
    cells_omega_in = mesh.locate_entities(domain, tdim, mark_omega_in)
    omega_in, _, _, _ = mesh.create_submesh(domain, tdim, cells_omega_in)

    # ### FE spaces
    degree = example.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(domain, fe) # full space
    W = fem.functionspace(omega_in, fe) # range space

    # ### Dirichlet dofs Gamma out
    # workaround: use dirichletbc to get correct dof indices
    zero = fem.Constant(domain, (default_scalar_type(0.), ) * gdim)
    gamma_dofs = fem.locate_dofs_geometrical(V, gamma_out)
    bc_gamma = fem.dirichletbc(zero, gamma_dofs, V)
    gamma_dofs = bc_gamma._cpp_object.dof_indices()[0]
    range_dofs = fem.locate_dofs_geometrical(V, mark_omega_in)
    bc_range = fem.dirichletbc(zero, range_dofs, V)
    range_dofs = bc_range._cpp_object.dof_indices()[0]

    # ### projection matrix (rigid body modes (rbm) in range space)
    product = InnerProduct(W, product="h1")
    product_mat = product.assemble_matrix()
    product = FenicsxMatrixOperator(product_mat, W, W)
    # consider using csr here
    M = product.matrix[:, :] # type: ignore
    nullspace = build_nullspace(FenicsxVectorSpace(W), product=product)
    R = nullspace.to_numpy().T
    right = np.dot(R.T, M)
    # middle = np.linalg.inv(np.dot(R.T, np.dot(M, R))) # should be the identity
    left = R
    # P = np.dot(left, np.dot(middle, right)) # projection matrix
    projection_matrix = np.dot(left, right)

    # ### Discretize A
    # A.assemble(mu) should yield same a as in non-parametric setting
    # no need to apply any boundary conditions
    from fom import ass_mat
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    num_subdomains = int(np.amax(cell_tags.values) + 1)
    matrices = []
    for id in range(num_subdomains):
        matrices.append(ass_mat(u, v, dx(id), [])) # note bcs=[]
    assert matrices[0] is not matrices[1]
    # wrap as pymor operator
    parameters = Parameters({"E": num_subdomains})
    parameter_space = ParameterSpace(parameters, (1., 2.)) # FIXME duplicate definition of parameter range
    parameter_functionals = [ProjectionParameterFunctional("E", size=num_subdomains, index=q) for q in range(num_subdomains)]
    ops = [FenicsxMatrixOperator(mat, V, V) for mat in matrices]
    operator = LincombOperator(ops, parameter_functionals)

    return {
            "A": operator,
            "parameter_space": parameter_space,
            "dirichlet_dofs": gamma_dofs,
            "range_dofs": range_dofs,
            "projection_matrix": projection_matrix
            }


if __name__ == "__main__":
    from tasks import ex
    rval = discretize_oversampling_problem(ex)
    # testset = rval["parameter_space"].sample_uniformly(2)
    # A = rval["A"]
    # mu1 = testset[0]
    # mu2 = testset[-1]
    # A1 = A.assemble(mu1)
    # A2 = A.assemble(mu2)
    # breakpoint()
