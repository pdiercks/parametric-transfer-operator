from typing import Tuple, Optional
import numpy as np
from scipy.sparse import coo_array

from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace


class COOMatrixOperator(Operator):
    """Wraps COO matrix data as an |Operator|.

    Args:
        data: The local reduced operator.
        rows: Global dof indices of non-zero row entries.
        cols: Global dof indices of non-zero cols entries.
        shape: The shape of the matrix.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    # TODO deal with boundary conditions
    # (i) apply BCs during copy from local to global? (requires to manipulate local data accordingly beforehand)
    # (ii) apply BCs to global matrix after assembly?

    def __init__(self, data: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: Tuple[int, int], solver_options: Optional[dict] = None, name: Optional[str] = None):
        self.__auto_init(locals()) # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)

        # TODO mu.to_numpy() is required?

        # TODO multiply data by mu
        # Approach A:
        # 1. Prepare data as list[np.ndarray] of length 10
        # 2. Multiply by mu_i and stack values in single array
        raise NotImplementedError
        data = self.data * mu # or similar
        rows = self.rows # type: ignore
        cols = self.cols # type: ignore
        assert data.shape == rows.shape
        assert cols.shape == rows.shape
        shape = self.shape # type: ignore

        K = coo_array((data, (rows, cols)), shape=shape)
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
