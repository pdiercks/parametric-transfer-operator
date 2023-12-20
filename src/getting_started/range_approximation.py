"""approximate the range of transfer operators for fixed parameter values"""

from time import perf_counter
# import concurrent.futures
from multi.transfer_operator import transfer_operator_subdomains_2d
from pymor.algorithms.rand_la import adaptive_rrf


def main():
    from tasks import beam
    from locmor import discretize_oversampling_problem
    disc = discretize_oversampling_problem(beam)
    parameter_space = disc.pop("parameter_space")
    A = disc.pop("A")
    dirichlet_dofs = disc.pop("dirichlet_dofs")
    target_dofs = disc.pop("target_dofs")
    P = disc.pop("projection_matrix")
    source_product = disc.pop("source_product")
    range_product = disc.pop("range_product")
    ntrain = 4
    trainset = parameter_space.sample_randomly(ntrain)

    # ### parallelization strategy
    # 1. mpirun --> MPI.COMM_SELF for mesh
    # 2. multiprocessing package
    # 3. pymor pool of workers (requirements by hapod?)
    # cannot use functions from pymor.algorithms.hapod directly
    # I don't want to compute a POD at each leaf node in the dist_hapod
    # The result of the range approx. is already an orthonormal basis

    def range_finder(mu):
        print(f"Approximating range of T for {mu=}")
        tic = perf_counter()
        raise NotImplementedError("Memory Issue, because P is not sparse!")
        T = transfer_operator_subdomains_2d(A.assemble(mu), dirichlet_dofs, target_dofs, projection_matrix=P)
        print(f"Build transfer operator of shape {T.matrix.shape} in {tic-perf_counter()}") # pyright: ignore[reportGeneralTypeIssues]
        ttol = 1e-3
        ftol = 1e-12
        num_testvecs = 20
        basis = adaptive_rrf(T, source_product=source_product, range_product=range_product,
                     tol=ttol, failure_tolerance=ftol, num_testvecs=num_testvecs)
        return basis


    basis = range_finder(trainset[0])
    # with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
    #     bases = executor.map(range_finder, trainset)

    # do POD over bases
    breakpoint()


if __name__ == "__main__":
    main()
