"""approximate the range of transfer operators for fixed parameter values"""

import os
from typing import Optional, Any
from itertools import repeat
import concurrent.futures
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv
from time import perf_counter

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.defaults import defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.constructions import NumpyConversionOperator
from pymor.parameters.base import Parameters, ParameterSpace

from multi.misc import x_dofs_vectorspace
from multi.problems import TransferProblem
from multi.sampling import correlation_matrix, create_random_values


@defaults('tol', 'failure_tolerance', 'num_testvecs')
def adaptive_rrf(
        T: TransferProblem, source_product=None, range_product=None, distribution: str = 'normal',
        sampling_options: Optional[dict[str, Any]] = None, tol: float = 1e-4,
        failure_tolerance: float = 1e-15, num_testvecs: int = 20, lambda_min: Optional[float] = None, iscomplex: bool = False):
    """Range approximation of transfer operator.

    Args:
        T: The associated transfer problem.
        source_product: Inner product for source space.
        range_product: Inner product for range space.
        distribution: The distribution to draw random samples from.
        sampling_options: Arguments for sampling method.
        tol: Target tolerance.
        failure_tolerance: Failure tolerance.
        num_testvecs: Number of vectors in the test set sampled from normal distribution.
        lambda_min: Min eigenvalue of source product.
        iscomplex: If True, use complex numbers.

    """
    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)
    assert isinstance(T, TransferProblem)

    B = T.range.empty()

    # always use normal distribution for test set
    R = T.generate_random_boundary_data(num_testvecs, distribution='normal')
    if iscomplex:
        R += 1j*T.generate_random_boundary_data(num_testvecs, distribution='normal')

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:
        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()

        def mvinv(v):
            return source_product.apply_inverse(source_product.range.from_numpy(v)).to_numpy()
        L = LinearOperator((source_product.source.dim, source_product.range.dim), matvec=mv)
        Linv = LinearOperator((source_product.range.dim, source_product.source.dim), matvec=mvinv)
        lambda_min = eigsh(L, sigma=0, which='LM', return_eigenvectors=False, k=1, OPinv=Linv)[0]

    testfail = failure_tolerance / min(T.source_gamma_out.dim, T.range.dim)
    testlimit = np.sqrt(2. * lambda_min) * erfinv(testfail**(1. / num_testvecs)) * tol
    maxnorm = np.inf
    M = T.solve(R)

    sampling_options = sampling_options or {}
    draw = True
    training_set = sampling_options.get('training_set')
    if training_set is not None:
        draw = False
    niter = 0
    while maxnorm > testlimit:
        basis_length = len(B)

        if draw:
            v = T.generate_random_boundary_data(1, distribution=distribution, **sampling_options)
            if iscomplex:
                v += 1j*T.generate_random_boundary_data(1, distribution=distribution, **sampling_options)
        else:
            assert training_set is not None
            v = training_set[np.newaxis, niter, :]

        B.append(T.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        niter += 1

    # ### convert to NumpyVectorArray
    # passing data along processes requires picklable data
    # fenics stuff is not picklable ...
    cop = NumpyConversionOperator(T.range, direction='to_numpy')
    B_numpy = cop.apply(B)

    return B_numpy


def build_mvn_training_set(transfer_problem):
    xmin = transfer_problem.problem.domain.xmin
    xmax = transfer_problem.problem.domain.xmax
    L_corr = np.linalg.norm(xmax - xmin).item() * 3

    x_dofs = x_dofs_vectorspace(transfer_problem.problem.V)
    points = x_dofs[transfer_problem.bc_dofs_gamma_out]

    num_samples = 0
    training_set = []
    while True:
        sigma = correlation_matrix(points, L_corr)
        print(f"Build Sigma of shape {sigma.shape} for {L_corr=}.")
        λ_max = eigsh(sigma, k=1, which="LM", return_eigenvectors=False)
        rtol = 5e-3
        eigvals = eigh(sigma, eigvals_only=True, driver='evx', subset_by_value=[λ_max.item() * rtol, np.inf])
        num_eigvals = eigvals.size
        print(f"Found {num_eigvals=}.")

        inc = num_eigvals - num_samples
        if inc > 0:
            mean = np.zeros(sigma.shape[0])
            u = create_random_values((inc, sigma.shape[0]), distribution='multivariate_normal', mean=mean, cov=sigma, method='eigh')
            training_set.append(u)

            num_samples += inc
            print(f"Added {inc=} samples")
            print("Decreasing correlation length ...")
            L_corr /= 2
        # elif inc == 0:
        #     # this means the correlation length was not decreased enough
        #     # but might happen for large L_corr in the beginning
        #     # How to differentiate between the above case and case when L_corr is already sufficiently small?
        #     L_corr /= 2
        else:
            break

    mvn_train_set = np.vstack(training_set)
    print(f"Build mvn training set of size {len(mvn_train_set)}")
    return mvn_train_set


def approximate_range(beam, mu, distribution='normal'):
    from .locmor import discretize_oversampling_problem

    logger = getLogger('range_approximation', level='INFO', filename=beam.range_approximation_log.as_posix())
    pid = os.getpid()
    logger.info(f"{pid=},\tApproximating range of T for {mu=} using {distribution=}.\n")

    tic = perf_counter()
    transfer_problem = discretize_oversampling_problem(beam, mu)
    logger.info(f"{pid=},\tDiscretized transfer problem in {perf_counter()-tic}.")

    if distribution == 'normal':
        sampling_options = {}
    elif distribution == 'multivariate_normal':
        mvn_train_set = build_mvn_training_set(transfer_problem)
        sampling_options = {'training_set': mvn_train_set}
    else:
        raise NotImplementedError

    ttol = beam.rrf_ttol
    ftol = beam.rrf_ftol
    num_testvecs = beam.rrf_num_testvecs
    source_product = transfer_problem.source_product
    range_product = transfer_problem.range_product
    # TODO approximate range in context of new_rng (if number of realizations > 1)
    basis = adaptive_rrf(
            transfer_problem, source_product=source_product, range_product=range_product,
            distribution=distribution, sampling_options=sampling_options, tol=ttol, failure_tolerance=ftol, num_testvecs=num_testvecs)
    logger.info(f"{pid=},\tNumber of basis functions is {len(basis)}.")

    # FenicsxMatrixOperator is not picklable
    product_matrix = csr_array(range_product.matrix.getValuesCSR()[::-1])
    product = NumpyMatrixOperator(product_matrix, source_id=basis.space.id, range_id=basis.space.id)
    return basis, product


def main(args):
    from .tasks import beam
    param = Parameters({"E": 3})
    parameter_space = ParameterSpace(param, beam.mu_range)
    # sufficient sampling? uniform?
    training_set = parameter_space.sample_randomly(args.ntrain)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        results = executor.map(approximate_range, repeat(beam), training_set, repeat(args.distribution))

    basis, range_product = next(results)
    snapshots = range_product.range.empty()
    snapshots.append(basis)
    for rb, _ in results:
        snapshots.append(rb)
    pod_modes, svals = pod(snapshots, product=range_product, rtol=beam.pod_rtol)

    breakpoint()

    # TODO write/define targets: pod_modes, svals, logfile(s)
    # TODO add a task to tasks.py
    # call this with command line arguments when called as module?
    # python3 -m src.getting_started.range_approximation arg1 arg2 works?


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("distribution", type=str, help="The distribution to draw samples from.")
    parser.add_argument("ntrain", type=int, help="The size of the training set.")
    parser.add_argument("--max_workers", type=int, default=4, help="The max number of workers.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
