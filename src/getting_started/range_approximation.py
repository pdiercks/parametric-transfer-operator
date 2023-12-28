"""approximate the range of transfer operators for fixed parameter values"""

from typing import Optional, Any
from time import perf_counter
import concurrent.futures
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import defaults
from pymor.operators.interface import Operator
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

    return B


def approximate_range(transfer_problem, distribution, sampling_options):
    ttol = 1e-3
    ftol = 1e-12
    num_testvecs = 20
    source_product = transfer_problem.source_product
    range_product = transfer_problem.range_product
    # TODO approximate range in context of new_rng (if number of realizations > 1)
    basis = adaptive_rrf(
            transfer_problem, source_product=source_product, range_product=range_product,
            distribution=distribution, sampling_options=sampling_options, tol=ttol, failure_tolerance=ftol, num_testvecs=num_testvecs)
    return basis


def main():
    from .tasks import beam
    from .locmor import discretize_oversampling_problem

    # TODO parameter space should only be defined once
    param = Parameters({"E": 3})
    parameter_space = ParameterSpace(param, (1., 2.))

    # fixed mu
    mu = parameter_space.sample_randomly(1)[0]

    print(f"Approximating range of T for {mu=}")
    tic = perf_counter()
    transfer_problem = discretize_oversampling_problem(beam, mu)
    print(f"Discretized transfer problem in {perf_counter()-tic}") # pyright: ignore[reportGeneralTypeIssues]

    # FIXME need access to transfer problem discretization data to define mvn_train_set
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

    tic = perf_counter()
    U = approximate_range(transfer_problem, 'normal', {})
    print(f"Approximated range using normal in {perf_counter()-tic}\n")

    tic = perf_counter()
    V = approximate_range(transfer_problem, 'multivariate_normal', {'training_set': mvn_train_set})
    print(f"Approximated range using multivariate normal in {perf_counter()-tic}\n")

    print(f"Length of U: {len(U)}")
    print(f"Length of V: {len(V)}")

    # ### parallelization strategy
    # 1. mpirun --> MPI.COMM_SELF for mesh
    # 2. multiprocessing package
    # 3. pymor pool of workers (requirements by hapod?)
    # cannot use functions from pymor.algorithms.hapod directly
    # I don't want to compute a POD at each leaf node in the dist_hapod
    # The result of the range approx. is already an orthonormal basis

    # TODO use ProcessPoolExecutor for parallelization
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     bases = executor.map(range_finder, trainset)

    # TODO do POD over all bases


if __name__ == "__main__":
    main()
