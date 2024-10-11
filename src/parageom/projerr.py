"""Compute projection error to assess quality of the basis."""

import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import orthogonal_part, project_array
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.parameters.base import ParameterSpace
from pymor.tools.random import new_rng
from scipy.stats import qmc


def main(args):
    from parageom.lhs import parameter_set
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory
    from parageom.tasks import example

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    logfilename = example.log_projerr(args.nreal, args.method, args.k, args.scale).as_posix()
    set_defaults({'pymor.core.logger.getLogger.filename': logfilename})
    logger = getLogger('projerr', level=loglevel)

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
    require_neumann_data = bool(np.any(np.nonzero(fext.to_numpy())[1]))
    if 0 in osp_config.cells_omega:
        assert require_neumann_data
    else:
        assert not require_neumann_data

    # ### Generate training seed for each of the 11 oversampling problems
    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    parameter_name = 'R'

    # check norm of UU and NN
    # UU = transfer.range.empty()
    # NN = transfer.range.empty()
    # theta = parameter_space.sample_randomly(5)
    # for mu in theta:
    #     transfer.assemble_operator(mu)
    #     R = transfer.generate_random_boundary_data(10)
    #     U = transfer.solve(R)
    #     UU.append(U)
    #     U_neumann = transfer.op.apply_inverse(fext)
    #     U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))
    #     U_orth = orthogonal_part(U_in_neumann, transfer.kernel, product=None, orthonormal=True)
    #     NN.append(U_orth)
    # breakpoint()

    myseeds_train = np.random.SeedSequence(example.projerr.seed_train).generate_state(11)
    ntrain = args.N
    dim = example.parameter_dim[args.k]
    sampler_train = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_train[args.k])
    training_set = parameter_set(sampler_train, ntrain, parameter_space, name=parameter_name)

    # seeds for the randomized range finder
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # ### Read basis and wrap as pymor object
    logger.info(f'Computing spectral basis with method {args.method} ...')
    epsilon_star = 0.01 * args.scale / example.energy_scale
    Nin = transfer.rhs.dofs.size
    basis = None
    svals = None
    sampling_options = {'scale': args.scale}

    if args.method == 'hapod':
        from parageom.hapod import adaptive_rrf_normal

        snapshots = transfer.range.empty()
        neumann_snapshots = transfer.range.empty(reserve=ntrain)
        spectral_basis_sizes = list()

        # use most conservative estimate on tolerances Nin=1
        epsilon_alpha = np.sqrt(Nin) * np.sqrt(1 - example.projerr.hapod_omega**2) * epsilon_star
        epsilon_pod = np.sqrt(Nin * ntrain) * example.projerr.hapod_omega * epsilon_star

        for mu, seed_seq in zip(training_set, seed_seqs_rrf):
            with new_rng(seed_seq):
                transfer.assemble_operator(mu)
                rb = adaptive_rrf_normal(
                    logger, transfer, num_testvecs=Nin, l2_err=epsilon_alpha, sampling_options=sampling_options
                )
                logger.info(f'\nSpectral Basis length: {len(rb)}.')
                spectral_basis_sizes.append(len(rb))
                snapshots.append(rb)

            if require_neumann_data:
                logger.info('\nSolving for additional Neumann mode ...')
                U_neumann = transfer.op.apply_inverse(fext)
                U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))  # type: ignore

                # ### Remove kernel after restriction to target subdomain
                if transfer.kernel is not None:
                    U_orth = orthogonal_part(
                        U_in_neumann,
                        transfer.kernel,
                        product=None,
                        orthonormal=True,
                    )
                else:
                    U_orth = U_in_neumann
                neumann_snapshots.append(U_orth)

        logger.info(f'Average length of spectral basis: {np.average(spectral_basis_sizes)}.')
        if require_neumann_data:
            logger.info(f'Number of Neumann snapshots: {len(neumann_snapshots)}.')
            snapshots.append(neumann_snapshots)

        logger.info('Subtracting ensemble mean ...')
        ensemble_mean = sum(snapshots)
        ensemble_mean.scal(1 / len(snapshots))
        snapshots.axpy(-1.0, ensemble_mean)

        logger.info('Computing final POD ...')
        basis, svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)

    elif args.method == 'hrrf':
        from parageom.heuristic import heuristic_range_finder

        # the training set used for training in the HAPOD
        # should be the same as the testing set in HRRF
        testing_set = training_set

        myseeds_hapod_train = np.random.SeedSequence(example.hrrf.seed_train).generate_state(11)
        sampler_test = qmc.LatinHypercube(dim, optimization='random-cd', seed=myseeds_hapod_train[args.k])
        hapod_training_set = parameter_set(sampler_test, args.ntrain_hrrf, parameter_space, name=parameter_name)

        with new_rng(seed_seqs_rrf[0]):
            spectral_basis = heuristic_range_finder(
                logger,
                transfer,
                sampler_train,
                parameter_space,
                hapod_training_set,
                testing_set,
                error_tol=example.hrrf.rrf_ttol / example.energy_scale,
                num_testvecs=example.hrrf.rrf_nt,
                block_size=args.bs,
                num_enrichments=example.hrrf.num_enrichments,
                radius_mu=example.hrrf.radius_mu,
                sampling_options=sampling_options,
                compute_neumann=require_neumann_data,
                fext=fext,
            )
        basis = spectral_basis
    else:
        raise NotImplementedError

    basis_length = len(basis)
    orthonormal = np.allclose(basis.gramian(transfer.range_product), np.eye(basis_length), atol=1e-5)
    if not orthonormal:
        raise ValueError('Basis is not orthonormal wrt range product.')

    # Definition of (random) test set (μ) and test data (g)
    size_test_set = args.num_samples * args.num_testvecs
    # if require_neumann_data:
    #     size_test_set += args.num_samples

    sampler_validation = qmc.LatinHypercube(dim, optimization='random-cd', seed=example.projerr.seed_test)
    validation_set = parameter_set(sampler_validation, args.num_samples, parameter_space, name=parameter_name)

    # ### test data / validation set
    test_data = transfer.range.empty(reserve=size_test_set)
    test_data_neumann = transfer.range.empty(reserve=args.num_samples)

    with logger.block(f'Computing test set of size {size_test_set}...'):
        with new_rng(example.projerr.seed_test // 2):
            for mu in validation_set:
                transfer.assemble_operator(mu)
                g = transfer.generate_random_boundary_data(args.num_testvecs, 'normal', sampling_options)
                test_data.append(transfer.solve(g))

                if require_neumann_data:
                    U_neumann = transfer.op.apply_inverse(fext)
                    U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))  # type: ignore

                    # ### Remove kernel after restriction to target subdomain
                    if transfer.kernel is not None:
                        U_orth = orthogonal_part(
                            U_in_neumann,
                            transfer.kernel,
                            product=None,
                            orthonormal=True,
                        )
                    else:
                        U_orth = U_in_neumann
                    test_data_neumann.append(U_orth)

    def compute_norm(U, key, value):
        if key == 'max':
            return U.amax()[1] * example.characteristic_displacement
        else:
            return U.norm(value) * example.energy_scale

    products = {transfer.range_product.name: transfer.range_product, 'max': False}
    test_norms = {}
    test_norms_neumann = {}
    error_types = ['relerr', 'abserr']
    numpy_funs = {'min': np.min, 'max': np.max, 'avg': np.average}
    output = {}
    outneumann = {}
    for product_name, product in products.items():
        test_norms[product_name] = compute_norm(test_data, product_name, product)
        test_norms_neumann[product_name] = compute_norm(test_data_neumann, product_name, product)
        output[f'l2_err_{product_name}'] = list()
        outneumann[f'l2_err_{product_name}'] = list()
        for etype in error_types:
            for fun in numpy_funs.keys():
                output[f'{fun}_{etype}_{product_name}'] = list()
                outneumann[f'{fun}_{etype}_{product_name}'] = list()

    logger.info('Computing projection error ...')
    for N in range(basis_length + 1):
        U_proj = project_array(
            test_data,
            basis[:N],
            product=transfer.range_product,
            orthonormal=orthonormal,
        )
        error = test_data - U_proj
        for k, v in products.items():
            error_norm = compute_norm(error, k, v)
            if np.all(error_norm == 0.0):
                # ensure to return 0 here even when the norm of U is zero
                rel_err = error_norm
            else:
                rel_err = error_norm / test_norms[k]
            l2_err = np.sum(error_norm**2.0) / size_test_set

            output[f'l2_err_{k}'].append(l2_err)
            for key, fun in numpy_funs.items():
                output[f'{key}_abserr_{k}'].append(fun(error_norm))
                output[f'{key}_relerr_{k}'].append(fun(rel_err))

    if require_neumann_data:
        logger.info('Computing projection error Neumann ...')
        for N in range(basis_length + 1):
            U_proj = project_array(
                test_data_neumann,
                basis[:N],
                product=transfer.range_product,
                orthonormal=orthonormal,
            )
            error = test_data_neumann - U_proj
            for k, v in products.items():
                error_norm = compute_norm(error, k, v)
                if np.all(error_norm == 0.0):
                    # ensure to return 0 here even when the norm of U is zero
                    rel_err = error_norm
                else:
                    rel_err = error_norm / test_norms_neumann[k]
                l2_err = np.sum(error_norm**2.0) / size_test_set

                outneumann[f'l2_err_{k}'].append(l2_err)
                for key, fun in numpy_funs.items():
                    outneumann[f'{key}_abserr_{k}'].append(fun(error_norm))
                    outneumann[f'{key}_relerr_{k}'].append(fun(rel_err))
        # append data to output
        for key, value in outneumann.items():
            output[f'neumann_{key}'] = value

    if args.show:
        import matplotlib.pyplot as plt

        plt.semilogy(
            np.arange(basis_length + 1),
            output[f'l2_err_{transfer.range_product.name}'],
            label='l2-mean, ' + transfer.range_product.name,
        )
        plt.semilogy(
            np.arange(basis_length + 1),
            output[f'max_relerr_{transfer.range_product.name}'],
            label='rel. err, ' + transfer.range_product.name,
        )
        plt.legend()
        plt.show()

    if args.output is not None:
        np.savez(
            args.output,
            **output,
            svals=svals if svals is not None else np.array([], dtype=np.float32),
        )


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        'Compute projection error over set of size `num_samples` * `num_testvecs`.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument('method', type=str, help='Method used for basis construction.', choices=('hapod', 'hrrf'))
    parser.add_argument('k', type=int, help='Use the k-th oversampling problem.')
    parser.add_argument(
        'N', type=int, help='Number of parameter samples in the training set (hapod), testing set (hrrf) respectively.'
    )
    parser.add_argument('num_samples', type=int, help='Number of parameters used to define the validation set.')
    parser.add_argument('num_testvecs', type=int, help='Number of test vectors used to define the validation set.')
    parser.add_argument(
        '--output',
        type=str,
        help='Write absolute and relative projection error to file.',
    )
    parser.add_argument(
        '--bs', type=int, help='Number of random samples per iteration in HRRF (block size).', default=1
    )
    parser.add_argument(
        '--ntrain_hrrf', type=int, help='Nmuber of initial parameter samples in the training set (HRRF).'
    )
    parser.add_argument('--scale', type=float, help='Value for scale in normal distribution.', default=0.1)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--show', action='store_true', help='Show projection error plot.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
