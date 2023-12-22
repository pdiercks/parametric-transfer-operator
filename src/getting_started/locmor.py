from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_array

from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Mu, Parameters, ParameterSpace

from multi.boundary import plane_at, within_range
from multi.domain import Domain, RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinElaSubProblem, TransferProblem
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


def discretize_oversampling_problem(example: Example, mu: Mu):
    """Returns TransferProblem for fixed parameter Mu.

    Args:
        example: The instance of the example dataclass.
        mu: The parameter value.

    """

    # use MPI.COMM_SELF for embarrassingly parallel workloads
    with XDMFFile(MPI.COMM_SELF, example.fine_oversampling_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, "subdomains")

    omega = Domain(domain, cell_tags=cell_tags)
    tdim = domain.topology.dim
    gdim = domain.ufl_cell().geometric_dimension()

    # ### Gamma out
    left = plane_at(omega.xmin[0], "x")
    right = plane_at(omega.xmax[0], "x")
    gamma_out = lambda x: np.logical_or(left(x), right(x))

    # ### Omega in
    mark_omega_in = within_range([1., 0.], [2., 1.])
    cells_omega_in = mesh.locate_entities(domain, tdim, mark_omega_in)
    omega_in, _, _, _ = mesh.create_submesh(domain, tdim, cells_omega_in)
    id_omega_in = 99
    omega_in = RectangularSubdomain(id_omega_in, omega_in)

    # ### FE spaces
    degree = example.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe) # full space
    W = fem.functionspace(omega_in.grid, fe) # range space

    # ### Oversampling problem
    E = example.youngs_modulus
    NU = example.poisson_ratio
    mu_values = mu.to_numpy()
    assert mu_values.size == 3
    materials = tuple([LinearElasticMaterial(gdim, E * mu_i, NU, plane_stress=False) for mu_i in mu_values])
    oversampling_problem = LinearElasticityProblem(omega, V, phases=materials)

    # ### Problem on target subdomain
    subproblem = LinElaSubProblem(omega_in, W, phases=(materials[1],))

    # ### TransferProblem
    transfer = TransferProblem(
            oversampling_problem,
            subproblem,
            gamma_out,
            dirichlet=None,
            source_product={"product": "l2"},
            range_product={"product": "h1"},
            remove_kernel=True,
            )
    return transfer




if __name__ == "__main__":
    from .tasks import beam
    param = Parameters({"E": 3})
    ps = ParameterSpace(param, (1., 2.))
    mu = ps.sample_randomly(1)[0]
    rval = discretize_oversampling_problem(beam, mu)
