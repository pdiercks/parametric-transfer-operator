from pathlib import Path

import numpy as np
from dolfinx.io import XDMFFile  # type: ignore
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import orthogonal_part
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVisualizer
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.parameters.base import ParameterSpace
from pymor.tools.random import new_rng
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv
from scipy.stats import qmc


# TODO: implement LHS for ParameterSpace?
# TODO: how to ensure that same mu's are not selected again as worst? --> track worst
# TODO: P contains new samples?
def enrich_training_set(sampler, parameter_space, training_set, testing_set, score, num_enrichments, radius):
    # score.shape = (ntest, numtesvecs)
    worst = np.argsort(score)[-num_enrichments:]
    samples = sampler.random(num_enrichments)
    dim = parameter_space.parameters['R']
    l_bounds = parameter_space.ranges['R'][:1] * dim
    u_bounds = parameter_space.ranges['R'][1:] * dim
    new_samples = qmc.scale(samples, l_bounds, u_bounds)

    # convert selected parameter values to numpy array
    parameters = [testing_set[i] for i in worst]
    refined_samples = []
    for mu in parameters:
        refined_samples.append(mu.to_numpy())
    refined_samples = np.array(refined_samples)
    new_samples = refined_samples + radius * new_samples

    # convert back to Mu and append to training set
    for x in new_samples:
        training_set.append(parameter_space.parameters.parse(x))


def heuristic_range_finder(
    logger,
    transfer_problem,
    sampler,
    parameter_space,
    training_set,
    testing_set,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    block_size: int = 1,
    num_enrichments: int = 5,
    radius_mu: float = 0.05,
    lambda_min=None,
    l2_err: float = 0.0,
    sampling_options=None,
    compute_neumann=True,
    fext=None,
):
    """Heuristic range approximation."""
    tp = transfer_problem
    distribution = 'normal'
    sampling_options = sampling_options or {}

    source_product = tp.source_product
    range_product = tp.range_product
    assert source_product is None or isinstance(source_product, Operator)
    assert range_product is None or isinstance(range_product, Operator)

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()  # type: ignore

        def mvinv(v):
            return source_product.apply_inverse(
                source_product.range.from_numpy(v)  # type: ignore
            ).to_numpy()

        L = LinearOperator(
            (source_product.source.dim, source_product.range.dim),  # type: ignore
            matvec=mv,  # type: ignore
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim),  # type: ignore
            matvec=mvinv,  # type: ignore
        )
        lambda_min = eigsh(L, sigma=0, which='LM', return_eigenvectors=False, k=1, OPinv=Linv)[0]

    logger.debug(f'Computing test set of size {len(testing_set) * num_testvecs} for spectral modes.')
    if compute_neumann:
        logger.debug(f'Computing test set of size {len(testing_set)} for neumann modes.')
        assert fext is not None

    M_s = tp.range.empty(reserve=len(testing_set) * num_testvecs)
    M_n = tp.range.empty(reserve=num_testvecs)  # global test set for neumann modes
    for mu in testing_set:
        tp.assemble_operator(mu)
        R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution, options=sampling_options)
        M_s.append(tp.solve(R))
        if compute_neumann:
            R_neumann = tp.op.apply_inverse(fext)
            R_in_neumann = tp.range.from_numpy(R_neumann.dofs(tp._restriction))
            M_n.append(orthogonal_part(R_in_neumann, tp.kernel, product=None, orthonormal=True))

    # ### Compute non-parametric testlimit
    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = tp.rhs.dofs.size
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    # use ntest instead of num_testvectors for the testlimit
    testlimit = np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / len(M_s))) * error_tol
    logger.info(f'{lambda_min=}')
    logger.info(f'{testlimit=}')

    B = tp.range.empty()
    l2_errors = None
    testlimit_l2 = None
    maxnorms = None
    M_s_norm2 = M_s.norm2(range_product)
    M_n_norm = None
    if compute_neumann:
        l2_errors = np.array([np.inf, np.inf], dtype=np.float64)
        maxnorms = np.array([np.inf, np.inf], dtype=np.float64)
        M_n_norm = M_n.norm2(range_product)
        testlimit_l2 = l2_err**2 * np.ones_like(l2_errors) / np.array([np.average(M_s_norm2), np.average(M_n_norm)])
    else:
        l2_errors = np.array([np.inf], dtype=np.float64)
        maxnorms = np.array([np.inf], dtype=np.float64)
        testlimit_l2 = l2_err**2 * np.ones_like(l2_errors) / np.array([np.average(M_s_norm2)])

    num_iter = 0
    num_neumann = 0
    enriched = 0
    while np.any(maxnorms > testlimit) and np.any(l2_errors > testlimit_l2):
        basis_length = len(B)
        ntrain = len(training_set)
        if num_iter > ntrain - 1:
            enrich_training_set(
                sampler,
                parameter_space,
                training_set,
                testing_set,
                np.max(M_s.norm(range_product).reshape(len(testing_set), num_testvecs), axis=1),
                num_enrichments,
                radius_mu,
            )
            enriched += 1
        mu = training_set[num_iter]
        tp.assemble_operator(mu)

        # add mode for spectral basis
        v = tp.generate_random_boundary_data(block_size, distribution, options=sampling_options)
        B.append(tp.solve(v))

        add_neumann = l2_errors[-1] > l2_err**2
        if compute_neumann and add_neumann:
            U_neumann = tp.op.apply_inverse(fext)
            U_in_neumann = tp.range.from_numpy(U_neumann.dofs(tp._restriction))
            U_orth = orthogonal_part(U_in_neumann, tp.kernel, product=None, orthonormal=True)
            B.append(U_orth)
            num_neumann += 1

        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)

        M_s -= B.lincomb(B.inner(M_s, range_product).T)
        maxnorms[0] = np.max(M_s.norm(range_product))
        l2_errors[0] = np.sum(M_s.norm2(range_product) / M_s_norm2) / len(M_s)

        if compute_neumann and add_neumann:
            M_n -= B.lincomb(B.inner(M_n, range_product).T)
            maxnorms[-1] = np.max(M_n.norm2(range_product))
            l2_errors[-1] = np.sum(M_n.norm2(range_product) / M_n_norm) / len(M_n)

        num_iter += 1
        logger.debug(f'{num_iter=}\t{maxnorms=}')
        logger.debug(f'{num_iter=}\t{l2_errors=}')

    reason = 'maxnorm' if np.all(maxnorms < testlimit) else 'l2err'
    logger.info(f'Had to compute {num_neumann} neumann modes.')
    logger.info(f'Had to enrich training set {enriched} times by {num_enrichments}.')
    logger.info(f'Finished heuristic range approx. in {num_iter} iterations ({reason=}).')

    return B


def parameter_set(sampler, num_samples, ps, name='R'):
    l_bounds = ps.ranges[name][:1] * ps.parameters[name]
    u_bounds = ps.ranges[name][1:] * ps.parameters[name]
    samples = qmc.scale(sampler.random(num_samples), l_bounds, u_bounds)
    s = []
    for x in samples:
        s.append(ps.parameters.parse(x))
    return s


def main(args):
    from parageom.lhs import sample_lhs
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory
    from parageom.tasks import example

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    method = Path(__file__).stem  # heuristic
    logfilename = example.log_basis_construction(args.nreal, method, args.k).as_posix()
    set_defaults({'pymor.core.logger.getLogger.filename': logfilename})
    logger = getLogger(method, level=loglevel)

    # ### Coarse grid partition of omega
    coarse_grid_path = example.path_omega_coarse(args.k)
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': example.gdim})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)

    # ### Fine grid partition of omega
    path_omega = example.path_omega(args.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega.as_posix(), 'r') as xdmf:
        omega_mesh = xdmf.read_mesh()
        omega_ct = xdmf.read_meshtags(omega_mesh, name='Cell tags')
        omega_ft = xdmf.read_meshtags(omega_mesh, name='mesh_tags')
    omega = RectangularDomain(omega_mesh, cell_tags=omega_ct, facet_tags=omega_ft)

    # ### Fine grid partition of omega_in
    path_omega_in = example.path_omega_in(args.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega_in.as_posix(), 'r') as xdmf:
        omega_in_mesh = xdmf.read_mesh()
    omega_in = RectangularDomain(omega_in_mesh)

    logger.info(f'Discretizing transfer problem for k = {args.k:02} ...')
    osp_config = oversampling_config_factory(args.k)
    transfer, fext = discretize_transfer_problem(example, struct_grid, omega, omega_in, osp_config, debug=args.debug)

    # ### Generate training seed for each of the 11 oversampling problems
    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    parameter_name = 'R'

    myseeds_train = np.random.SeedSequence(example.training_set_seed).generate_state(11)
    ntrain = example.ntrain('heuristic', args.k)
    dim = example.parameter_dim[args.k]
    sampler_train = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_train[args.k])
    training_set = parameter_set(sampler_train, ntrain, parameter_space, name=parameter_name)

    # do the same for testing set
    myseeds_test = np.random.SeedSequence(example.testing_set_seed).generate_state(11)
    sampler_test = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_test[args.k])
    ntest = example.ntest('heuristic', args.k)
    testing_set = parameter_set(sampler_test, ntest, parameter_space, name=parameter_name)

    logger.info(
        'Starting Heuristic Randomized Range Approximation of parametric transfer operator'
        f' for training set of size {len(training_set)}.'
    )

    # ### Generate random seed to draw random samples in the range finder algorithm
    # in the case of the heuristic range finder there is only one while loop
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(1)

    require_neumann_data = bool(np.any(np.nonzero(fext.to_numpy())[1]))
    if 0 in osp_config.cells_omega:
        assert require_neumann_data
    else:
        assert not require_neumann_data

    # ### Heuristic randomized range finder
    epsilon_star = example.epsilon_star['heuristic']

    logger.debug(f'{seed_seqs_rrf[0]=}')
    with new_rng(seed_seqs_rrf[0]):
        spectral_basis = heuristic_range_finder(
            logger,
            transfer,
            sampler_train,
            parameter_space,
            training_set,
            testing_set,
            error_tol=example.rrf_ttol,
            failure_tolerance=example.rrf_ftol,
            num_testvecs=example.rrf_num_testvecs,
            l2_err=epsilon_star,
            compute_neumann=require_neumann_data,
            fext=fext,
        )

    # ### Compute Neumann Modes and extend basis
    basis_length = len(spectral_basis)

    if args.debug:
        assert np.allclose(spectral_basis.gramian(transfer.range_product), np.eye(len(spectral_basis)))

    logger.info(f'Final basis length (k={args.k:02}): {basis_length}.')

    if logger.level == 10:  # DEBUG
        viz = FenicsxVisualizer(spectral_basis.space)
        viz.visualize(spectral_basis, filename=example.heuristic_modes_xdmf(args.nreal, args.k))
    np.save(
        example.heuristic_modes_npy(args.nreal, args.k),
        spectral_basis.to_numpy(),
    )


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Oversampling for ParaGeom examples using Heuristic Randomized Range Finder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('nreal', type=int, help='The n-th realization of the problem.')
    parser.add_argument('k', type=int, help='The oversampling problem for target subdomain Ω_in^k.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
