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
    # TODO new_samples may not be within bounds
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
    num_testvecs: int = 20,
    block_size: int = 1,
    num_enrichments: int = 5,
    radius_mu: float = 0.05,
    sampling_options=None,
    compute_neumann=False,
    fext=None,
):
    """Heuristic range approximation."""
    tp = transfer_problem
    sampling_options = sampling_options or {}

    range_product = tp.range_product
    assert range_product is None or isinstance(range_product, Operator)

    logger.debug(f'Computing test set of size {len(testing_set) * num_testvecs} for spectral modes.')
    if compute_neumann:
        logger.debug(f'Computing test set of size {len(testing_set)} for neumann modes.')
        assert fext is not None

    # testvector sets
    M_s = tp.range.empty(reserve=len(testing_set) * num_testvecs)
    M_n = tp.range.empty(reserve=len(testing_set))
    for mu in testing_set:
        tp.assemble_operator(mu)
        R = tp.generate_random_boundary_data(num_testvecs, 'normal', sampling_options)
        M_s.append(tp.solve(R))
        if compute_neumann:
            R_neumann = tp.op.apply_inverse(fext)
            R_in_neumann = tp.range.from_numpy(R_neumann.dofs(tp._restriction))
            M_n.append(orthogonal_part(R_in_neumann, tp.kernel, product=None, orthonormal=True))

    # initial maxnorm of testvectors
    maxnorms0_s = M_s.norm(range_product)
    maxnorms0_n = None
    maxnorm = np.array([np.inf], dtype=np.float64)
    if compute_neumann:
        maxnorms0_n = M_n.norm(range_product)
        maxnorm = np.append(maxnorm, np.inf)

    B = tp.range.empty()
    num_iter = 0
    num_neumann = 0
    enriched = 0
    while np.any(maxnorm > error_tol):
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
        v = tp.generate_random_boundary_data(block_size, 'normal', sampling_options)
        B.append(tp.solve(v))

        add_neumann = maxnorm[-1] > error_tol
        if compute_neumann and add_neumann:
            U_neumann = tp.op.apply_inverse(fext)
            U_in_neumann = tp.range.from_numpy(U_neumann.dofs(tp._restriction))
            U_orth = orthogonal_part(U_in_neumann, tp.kernel, product=None, orthonormal=True)
            B.append(U_orth)
            num_neumann += 1

        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)

        M_s -= B.lincomb(B.inner(M_s, range_product).T)
        maxnorm[0] = np.max(M_s.norm(range_product) / maxnorms0_s)

        if compute_neumann and add_neumann:
            M_n -= B.lincomb(B.inner(M_n, range_product).T)
            maxnorm[-1] = np.max(M_n.norm(range_product) / maxnorms0_n)

        num_iter += 1
        logger.debug(f'{num_iter=}\t{maxnorm=}')

    logger.info(f'Had to compute {num_neumann} neumann modes.')
    logger.info(f'Had to enrich training set {enriched} times by {num_enrichments}.')
    logger.info(f'Finished heuristic range approx. in {num_iter} iterations.')

    return B


def main(args):
    from parageom.lhs import parameter_set
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory
    from parageom.tasks import example

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    method = 'hrrf'
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

    myseeds_train = np.random.SeedSequence(example.hrrf.seed_train).generate_state(11)
    dim = example.parameter_dim[args.k]
    ntrain = example.hrrf.ntrain(dim)
    sampler_train = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_train[args.k])
    training_set = parameter_set(sampler_train, ntrain, parameter_space, name=parameter_name)

    # debug
    # check norm of u
    # UU = transfer.range.empty()
    # NN = transfer.range.empty()
    # for mu in training_set[:10]:
    #     transfer.assemble_operator(mu)
    #     R = transfer.generate_random_boundary_data(1, 'normal', {'scale': 0.1})
    #     U = transfer.solve(R)
    #     UU.append(U)
    #
    #     U_neumann = transfer.op.apply_inverse(fext)
    #     U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))
    #     U_orth = orthogonal_part(U_in_neumann, transfer.kernel, product=None, orthonormal=True)
    #     NN.append(U_orth)
    # breakpoint()

    # do the same for testing set
    myseeds_test = np.random.SeedSequence(example.hrrf.seed_test).generate_state(11)
    sampler_test = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_test[args.k])
    ntest = example.hrrf.ntest(dim)
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
    sampling_options = {'scale': example.g_scale}
    logger.debug(f'{seed_seqs_rrf[0]=}')
    with new_rng(seed_seqs_rrf[0]):
        spectral_basis = heuristic_range_finder(
            logger,
            transfer,
            sampler_train,
            parameter_space,
            training_set,
            testing_set,
            error_tol=example.hrrf.rrf_ttol / example.energy_scale,
            num_testvecs=example.hrrf.rrf_nt,
            block_size=1,
            num_enrichments=example.hrrf.num_enrichments,
            radius_mu=example.hrrf.radius_mu,
            sampling_options=sampling_options,
            compute_neumann=require_neumann_data,
            fext=fext,
        )

    # ### Compute Neumann Modes and extend basis
    basis_length = len(spectral_basis)

    if args.debug:
        assert np.allclose(spectral_basis.gramian(transfer.range_product), np.eye(len(spectral_basis)))

    logger.info(f'Final basis length (k={args.k:02}): {basis_length}.')

    if args.debug:
        viz = FenicsxVisualizer(spectral_basis.space)
        viz.visualize(spectral_basis, filename=example.modes_xdmf('hrrf', args.nreal, args.k))
    np.save(
        example.modes_npy('hrrf', args.nreal, args.k),
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
