"""post-processing of stress"""

from collections import namedtuple

import basix
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from multi.domain import StructuredQuadGrid
from multi.io import read_mesh
from pymor.core.pickle import load

StressWrapper = namedtuple('StressWrapper', ['expr', 'cells', 'fun'])
# TODO Re-evaluate design of FOM and ROM (and ModelWrapper)
ModelWrapper = namedtuple(
    'ModelWrapper', ['model', 'solution', 'dofmap', 'modes', 'sol_local', 'aux', 'trafo'], defaults=(None,) * 5
)


def main(args):
    """Compute principal stress using FOM and ROM."""
    from parageom.dofmap_gfem import GFEMDofMap
    from parageom.fom import ParaGeomLinEla
    from parageom.locmor import reconstruct
    from parageom.tasks import example
    from parageom.validate_rom import build_fom, build_rom

    # ### FOM
    fom, parageom_fom = build_fom(example, ω=args.omega)

    # ### Function for displacement on unit cell (for reconstruction)
    unit_cell_domain = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={'gdim': example.gdim})[0]
    V_i = df.fem.functionspace(unit_cell_domain, ('P', example.fe_deg, (example.gdim,)))
    u_local = df.fem.Function(V_i, name='u_i')

    # ### Build localized ROM
    coarse_grid_path = example.coarse_grid
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': 2})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)
    dofmap = GFEMDofMap(struct_grid)
    params = fom.parameters
    num_modes = 100
    rom, selected_modes, aux = build_rom(
        example, dofmap, params, num_modes, ω=args.omega, nreal=0, method=args.method, use_ei=True
    )

    # ### Global Function for displacement solution
    V = fom.solution_space.V
    u_fom = df.fem.Function(V, name='u_fom')
    u_rom = df.fem.Function(V, name='u_rom')
    d_rom = df.fem.Function(V, name='d_rom')  # global transformation displacement

    def compute_principal_components(f, num_cells):
        values = f.reshape(num_cells, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx + fyy) / 2 - np.sqrt(((fxx - fyy) / 2) ** 2 + fxy**2)
        fmax = (fxx + fyy) / 2 + np.sqrt(((fxx - fyy) / 2) ** 2 + fxy**2)
        return fmin, fmax

    # ### Quadrature space for stress
    # Note that the function is defined on the submesh, such that fun.x.array has correct size
    # The reduced evaluation is achieved by passing cells=cell_map
    omegagl = fom.solution_space.V.mesh
    tdim = omegagl.topology.dim
    map_c = omegagl.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    basix_celltype = getattr(basix.CellType, omegagl.topology.cell_type.name)
    q_degree = 2
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    qve = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme='default', degree=q_degree)  # type: ignore
    QV = df.fem.functionspace(omegagl, qve)

    stress_fom = df.fem.Function(QV)
    stress_rom = df.fem.Function(QV)

    # ### UFL representation and Expression of stress for both models
    suf = parageom_fom.weighted_stress(u_fom)
    stress_ufl_fom_vector = ufl.as_vector([suf[0, 0], suf[1, 1], suf[2, 2], suf[0, 1]])
    stress_expr_fom = df.fem.Expression(stress_ufl_fom_vector, q_points)

    rommat = {
        'gdim': parageom_fom.domain.gdim,
        'E': example.E,
        'NU': example.NU,
        'plane_stress': example.plane_stress,
    }
    parageom_rom = ParaGeomLinEla(parageom_fom.domain, V, d_rom, rommat)
    sur = parageom_rom.weighted_stress(u_rom)
    stress_ufl_rom_vector = ufl.as_vector([sur[0, 0], sur[1, 1], sur[2, 2], sur[0, 1]])
    stress_expr_rom = df.fem.Expression(stress_ufl_rom_vector, q_points)

    # ### Finally wrap stress objects
    SigmaFom = StressWrapper(stress_expr_fom, cells, stress_fom)
    SigmaRom = StressWrapper(stress_expr_rom, cells, stress_rom)

    # Wrap models
    wrapped_fom = ModelWrapper(fom, u_fom)
    wrapped_rom = ModelWrapper(rom, u_rom, dofmap, selected_modes, u_local, aux, d_rom)

    def evaluate_constraint(wrapped_model, mu, stress_expr, cells, stress_fun):
        model = wrapped_model.model
        if model.name == 'FOM':
            # in the case of the FOM, d_trafo is updated before solve
            U = model.solve(mu)
            wrapped_model.solution.x.array[:] = U.to_numpy().flatten()
        else:
            urb = model.solve(mu)
            # reconstruct updates global displacement solution and
            # transformation displacement
            reconstruct(
                urb.to_numpy(),
                mu,
                wrapped_model.dofmap,
                wrapped_model.modes,
                wrapped_model.sol_local.function_space,
                wrapped_model.solution,
                wrapped_model.trafo,
                wrapped_model.aux,
            )

        # now that u & d are uptodate, compute stress
        values = stress_fun.x.array.reshape(cells.size, -1)
        stress_expr.eval(V.mesh, entities=cells, values=values)
        s1, s2 = compute_principal_components(values, cells.size)

        # TODO: scaling and stress limit
        upper_bound = 2.2 * example.sigma_scale
        lower_bound = 30 * example.sigma_scale
        return s1 / lower_bound, s2 / upper_bound

    # ### Get optimal solution μ*
    def ReadOptima(example):
        rv = {}
        d = {'fom': example.fom_minimization_data('hrrf', 0), 'rom': example.rom_minimization_data('hrrf', 0)}
        for k, v in d.items():
            with v.open('rb') as fh:
                data = load(fh)
            try:
                mu = fom.parameters.parse(data['mu_min'])
            except KeyError:
                mu = fom.parameters.parse(data['mu_N_min'])
            rv[k] = mu
        return rv

    mu_star = ReadOptima(example)
    s1_fom, s2_fom = evaluate_constraint(wrapped_fom, mu_star['fom'], stress_expr_fom, cells, stress_fom)
    s1_rom, s2_rom = evaluate_constraint(wrapped_rom, mu_star['rom'], stress_expr_rom, cells, stress_rom)

    # targets
    from parageom.stress_analysis import project

    xdmf_files = example.pp_stress(args.method, nr=0)

    ## Output Quadrature space
    qs = basix.ufl.quadrature_element(basix_celltype, value_shape=(2,), scheme='default', degree=q_degree)
    Q = df.fem.functionspace(fom.solution_space.V.mesh, qs)
    sigma_q = df.fem.Function(Q, name='sp')
    W = df.fem.functionspace(fom.solution_space.V.mesh, ('P', example.fe_deg, (2,)))  # output space
    sigma_p = df.fem.Function(W, name='sp')

    sigma_q.x.array[::2] = s1_fom.flatten()
    sigma_q.x.array[1::2] = s2_fom.flatten()
    project(sigma_q, sigma_p)

    def translate_mesh(x, dtrafo, reverse=False):
        delta_x = np.pad(dtrafo.x.array.reshape(x.shape[0], -1), pad_width=[(0, 0), (0, 1)])
        if reverse:
            x -= delta_x
        else:
            x += delta_x

    # Stress FOM
    translate_mesh(Q.mesh.geometry.x, parageom_fom.d)

    with df.io.XDMFFile(Q.mesh.comm, xdmf_files['fom'].as_posix(), 'w') as xdmf:  # type: ignore
        xdmf.write_mesh(Q.mesh)
        xdmf.write_function(sigma_p)  # type: ignore

    translate_mesh(Q.mesh.geometry.x, parageom_fom.d, reverse=True)

    # Stress ROM
    sigma_q.x.array[::2] = s1_rom.flatten()
    sigma_q.x.array[1::2] = s2_rom.flatten()
    project(sigma_q, sigma_p)

    translate_mesh(Q.mesh.geometry.x, d_rom)

    with df.io.XDMFFile(Q.mesh.comm, xdmf_files['rom'].as_posix(), 'w') as xdmf:  # type: ignore
        xdmf.write_mesh(Q.mesh)
        xdmf.write_function(sigma_p)  # type: ignore

    translate_mesh(Q.mesh.geometry.x, d_rom, reverse=True)

    # Stres Error
    sigma_q.x.array[::2] = np.abs(s1_fom - s1_rom).flatten()
    sigma_q.x.array[1::2] = np.abs(s2_fom - s2_rom).flatten()
    project(sigma_q, sigma_p)
    with df.io.XDMFFile(Q.mesh.comm, xdmf_files['err'].as_posix(), 'w') as xdmf:  # type: ignore
        xdmf.write_mesh(Q.mesh)
        xdmf.write_function(sigma_p)  # type: ignore


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Compute principal stress using FOM and localized ROM.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'method',
        type=str,
        help='The name of the training strategy.',
        choices=('hapod', 'hrrf'),
    )
    parser.add_argument('--omega', type=float, help='Weighting factor for output functional.', default=0.5)
    args = parser.parse_args(sys.argv[1:])
    main(args)
