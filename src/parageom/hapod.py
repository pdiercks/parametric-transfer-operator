import numpy as np
from dolfinx.io import XDMFFile  # type: ignore
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import orthogonal_part
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.core.pickle import dump
from pymor.operators.interface import Operator
from pymor.parameters.base import ParameterSpace
from pymor.tools.random import new_rng
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.stats import qmc


def adaptive_rrf_normal(
    logger,
    transfer_problem,
    num_testvecs: int = 20,
    lambda_min=None,
    l2_err: float = 0.0,
    sampling_options=None,
):
    r"""Adaptive randomized range approximation of `A`."""
    tp = transfer_problem
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

    R = tp.generate_random_boundary_data(num_testvecs, 'normal', sampling_options)
    M = tp.solve(R)
    B = tp.range.empty()
    l2 = np.inf

    while l2 > l2_err**2.0:
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, 'normal', sampling_options)
        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        l2 = np.sum(M.norm2(range_product))
        logger.debug(f'{l2=}')

    logger.info(f'{l2 < l2_err ** 2 =}\t{l2=}')
    logger.info(f'Finished RRF in {len(B)} iterations.')

    return B


def main(args):
    from parageom.lhs import parameter_set
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory
    from parageom.tasks import example

    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    logfilename = example.log_basis_construction(args.nreal, 'hapod', args.k).as_posix()
    set_defaults({'pymor.core.logger.getLogger.filename': logfilename})
    logger = getLogger('hapod', level=loglevel)

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
    myseeds = np.random.SeedSequence(example.hapod.seed_train).generate_state(11)
    pdim = example.parameter_dim[args.k]
    sampler_train = qmc.LatinHypercube(pdim, optimization='random-cd', seed=myseeds[args.k])
    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    parameter_name = 'R'
    ntrain = example.hapod.ntrain(pdim)
    training_set = parameter_set(sampler_train, ntrain, parameter_space, name=parameter_name)
    logger.info('Starting range approximation of transfer operators' f' for training set of size {len(training_set)}.')

    # ### Generate random seed for each specific mu in the training set
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # start range approximation with `transfer` and `F_ext`
    require_neumann_data = np.any(np.nonzero(fext.to_numpy())[1])
    if 0 in osp_config.cells_omega:
        assert require_neumann_data
    else:
        assert not require_neumann_data

    assert len(training_set) == len(seed_seqs_rrf)
    snapshots = transfer.range.empty()
    neumann_snapshots = transfer.range.empty(reserve=len(training_set))
    spectral_basis_sizes = list()

    epsilon_star = example.hapod.eps / example.energy_scale
    Nin = transfer.rhs.dofs.size
    epsilon_alpha = np.sqrt(Nin) * np.sqrt(1 - example.hapod.omega**2.0) * epsilon_star
    epsilon_pod = np.sqrt(Nin * ntrain) * example.hapod.omega * epsilon_star

    sampling_options = {'scale': example.g_scale}
    for mu, seed_seq in zip(training_set, seed_seqs_rrf):
        with new_rng(seed_seq):
            transfer.assemble_operator(mu)
            basis = adaptive_rrf_normal(
                logger, transfer, num_testvecs=Nin, l2_err=epsilon_alpha, sampling_options=sampling_options
            )
            logger.info(f'\nSpectral Basis length: {len(basis)}.')
            spectral_basis_sizes.append(len(basis))
            snapshots.append(basis)  # type: ignore

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
                neumann_snapshots.append(U_orth)  # type: ignore

    logger.info(f'Average length of spectral basis: {np.average(spectral_basis_sizes)}.')
    if len(neumann_snapshots) > 0:  # type: ignore
        logger.info('Appending Neumann snapshots to global snapshot set.')
        snapshots.append(neumann_snapshots)  # type: ignore

    logger.info('Subtracting ensemble mean ...')
    ensemble_mean = sum(snapshots)
    ensemble_mean.scal(1 / len(snapshots))
    snapshots.axpy(-1.0, ensemble_mean)

    logger.info('Computing final POD')
    spectral_modes, spectral_svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)  # type: ignore

    if logger.level == 10:  # DEBUG
        from pymor.bindings.fenicsx import FenicsxVisualizer

        viz = FenicsxVisualizer(transfer.range)
        hapod_modes_xdmf = example.hapod_modes_xdmf(args.nreal, args.k).as_posix()
        viz.visualize(spectral_modes, filename=hapod_modes_xdmf)

    np.save(
        example.hapod_modes_npy(args.nreal, args.k),
        spectral_modes.to_numpy(),
    )
    np.save(
        example.hapod_singular_values(args.nreal, args.k),
        spectral_svals,
    )
    hapod_info = {
        'epsilon_star': epsilon_star,
        'epsilon_alpha': epsilon_alpha,
        'epsilon_pod': epsilon_pod,
        'avg_basis_length': np.average(spectral_basis_sizes),
        'num_snapshots': len(snapshots),
        'num_modes': len(spectral_modes),
    }
    with example.hapod_info(args.nreal, args.k).open('wb') as fh:
        dump(hapod_info, fh)


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Oversampling for ParaGeom example using HAPOD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('nreal', type=int, help='The n-th realization of the problem.')
    parser.add_argument('k', type=int, help='The oversampling problem for target subdomain Ω_in^k.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
