import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.projection import orthogonal_part
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVisualizer
from pymor.parameters.base import ParameterSpace


def main(args):
    from parageom.locmor import discretize_transfer_problem, oversampling_config_factory
    from parageom.tasks import example

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

    osp_config = oversampling_config_factory(args.k)
    transfer, fext = discretize_transfer_problem(example, struct_grid, omega, omega_in, osp_config, debug=args.debug)
    parameter_space = ParameterSpace(transfer.operator.parameters, example.mu_range)
    U = transfer.range.empty()
    V = transfer.range.empty()
    theta = parameter_space.sample_randomly(20)

    for mu in theta:
        transfer.assemble_operator(mu)

        R = transfer.generate_random_boundary_data(1, 'normal', options={'scale': example.characteristic_displacement})
        U.append(transfer.solve(R))

        U_neumann = transfer.op.apply_inverse(fext)
        U_in_neumann = transfer.range.from_numpy(U_neumann.dofs(transfer._restriction))
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
        V.append(U_orth)

    # unorm = U.norm(transfer.range_product) * example.energy_scale
    # vnorm = V.norm(transfer.range_product) * example.energy_scale
    # pinfo = lambda x: print((np.min(x), np.average(x), np.max(x)))
    B = gram_schmidt(U, product=transfer.range_product, copy=True)

    if args.debug:
        viz = FenicsxVisualizer(B.space)
        viz.visualize(B, filename='output/test_transfer.xdmf')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        'Test transfer operator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('k', type=int, help='Use the k-th oversampling problem.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
