import pathlib
from time import perf_counter

import basix
import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.io import read_mesh
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.models.basic import StationaryModel
from pymor.operators.constructions import ConstantOperator, LincombOperator, VectorFunctional
from pymor.parameters.base import Parameters
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.tools.random import new_rng
from pymor.vectorarrays.numpy import NumpyVectorSpace


def main(args):
    from parageom.dofmap_gfem import GFEMDofMap
    from parageom.fom import ParaGeomLinEla
    from parageom.locmor import assemble_gfem_system, reconstruct
    from parageom.tasks import example

    stem = pathlib.Path(__file__).stem
    logfilename = example.log_validate_rom(args.nreal, args.num_modes, method=args.method, ei=args.ei).as_posix()
    set_defaults({'pymor.core.logger.getLogger.filename': logfilename})
    if args.debug:
        loglevel = 10  # debug
    else:
        loglevel = 20  # info
    logger = getLogger(stem, level=loglevel)

    # ### FOM
    fom, parageom_fom = build_fom(example)
    V = fom.solution_space.V

    # ### Global Functions
    u_rom = df.fem.Function(V, name='urom')  # FOM displacement
    u_fom = df.fem.Function(V, name='ufom')  # ROM displacement
    d_rom = df.fem.Function(V, name='drom')  # ROM transformation displacement
    # FOM transformation displacement is managed by `parageom_fom` object

    def constrained_cells(domain):
        """Get active cells to deactivate constraint function in some part of the domain (near the support)."""

        def exclude(x):
            radius = 0.3
            center = np.array([[10.0], [0.0], [0.0]])
            distance = np.linalg.norm(np.abs(x - center), axis=0)
            return distance < radius

        tdim = domain.topology.dim
        map_c = domain.topology.index_map(tdim)
        num_cells = map_c.size_local + map_c.num_ghosts
        allcells = np.arange(0, num_cells, dtype=np.int32)

        nonactive = df.mesh.locate_entities(domain, tdim, exclude)
        active = np.setdiff1d(allcells, nonactive)
        return active

    constrained = constrained_cells(V.mesh)
    submesh, cell_map, _, _ = df.mesh.create_submesh(V.mesh, V.mesh.topology.dim, constrained)

    # ### Quadrature space for stress
    basix_celltype = getattr(basix.CellType, submesh.topology.cell_type.name)
    q_degree = 2
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    qve = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme='default', degree=q_degree)
    QV = df.fem.functionspace(submesh, qve)

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

    # tdim = V.mesh.topology.dim
    # map_c = V.mesh.topology.index_map(tdim)
    # num_cells = map_c.size_local + map_c.num_ghosts
    # cells = np.arange(0, num_cells, dtype=np.int32)

    # ### NumpyVectorSpace for Stress
    num_cells = cell_map.size
    num_qp = q_points.shape[0]
    dim_stress_space = num_qp * num_cells
    stress_space = NumpyVectorSpace(dim_stress_space)

    def compute_first_principal(f, num_cells):
        values = f.reshape(num_cells, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmax = (fxx + fyy) / 2 + np.sqrt(((fxx - fyy) / 2) ** 2 + fxy**2)
        return fmax.flatten()

    # ### Function for displacement on unit cell (for reconstruction)
    unit_cell_domain = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={'gdim': example.gdim})[0]
    V_i = df.fem.functionspace(unit_cell_domain, ('P', example.fe_deg, (example.gdim,)))
    u_local = df.fem.Function(V_i, name='u_i')

    # ### Build localized ROM
    coarse_grid_path = example.coarse_grid
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': example.gdim})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)
    dofmap = GFEMDofMap(struct_grid)
    params = fom.parameters
    num_modes = args.num_modes

    rom = None
    if args.ei:
        logger.info('Building ROM with EI ...')
        tic = perf_counter()
        rom, modes, auxmodel = build_rom(
            example,
            dofmap,
            params,
            num_modes,
            ω=args.omega,
            nreal=args.nreal,
            method=args.method,
            use_ei=args.ei,
        )
        logger.info(f'Took {perf_counter()-tic} to build ROM.')
        rom_data = {}
    else:
        logger.info('Building ROM without EI ...')
        # here time is not interesting as the assembly has to be carried out
        # every time model is evaluated for new `mu`
        rom_data, auxmodel = build_rom(
            example,
            dofmap,
            params,
            num_modes,
            ω=args.omega,
            nreal=args.nreal,
            method=args.method,
            use_ei=args.ei,
        )

    P = params.space(example.mu_range)
    with new_rng(example.rom_validation.seed):
        validation_set = P.sample_randomly(args.num_params)

    ufom_sols = fom.solution_space.empty()
    urom_sols = fom.solution_space.empty()
    dfom_sols = fom.solution_space.empty()
    drom_sols = fom.solution_space.empty()
    sfom_sols = stress_space.empty()
    srom_sols = stress_space.empty()

    energy_product = fom.products['energy']
    kappa = np.empty(len(validation_set), dtype=np.float64) if args.condition else None

    for i_mu, mu in enumerate(validation_set):
        U_fom = fom.solve(mu)
        ufom_sols.append(U_fom)
        u_fom.x.array[:] = U_fom.to_numpy().flatten()  # type: ignore
        dfom_sols.append(fom.solution_space.make_array([parageom_fom.d.x.petsc_vec.copy()]))

        if args.ei:
            assert rom is not None
            urb = rom.solve(mu)
        else:
            operator, rhs, modes = assemble_gfem_system(
                rom_data['dofmap'],
                rom_data['operator_local'],
                rom_data['rhs_local'],
                mu,
                rom_data['local_bases'],
                rom_data['dofs_per_vert'],
                rom_data['max_dofs_per_vert'],
            )
            rom = StationaryModel(operator, rhs, output_functional=None, name='ROM')
            urb = rom.solve(mu)
        reconstruct(
            urb.to_numpy(),
            mu,
            dofmap,
            modes,
            u_local.function_space,
            u_rom,
            d_rom,
            auxmodel,
        )
        U_rom = fom.solution_space.make_array([u_rom.x.petsc_vec.copy()])  # type: ignore
        urom_sols.append(U_rom)
        drom_sols.append(fom.solution_space.make_array([d_rom.x.petsc_vec.copy()]))

        if args.condition:
            A = rom.operator.assemble(mu)
            if A.sparse:
                kappa[i_mu] = np.linalg.cond(A.matrix.todense())
            else:
                kappa[i_mu] = np.linalg.cond(A.matrix)

        stress_expr_rom.eval(V.mesh, entities=cell_map, values=stress_rom.x.array.reshape(cell_map.size, -1))
        s_rom = compute_first_principal(stress_rom.x.array, cell_map.size)
        srom_sols.append(stress_space.make_array(s_rom))

        stress_expr_fom.eval(V.mesh, entities=cell_map, values=stress_fom.x.array.reshape(cell_map.size, -1))
        s_fom = compute_first_principal(stress_fom.x.array, cell_map.size)
        sfom_sols.append(stress_space.make_array(s_fom))

        # TODO: add option --pvplot, s.t. for worst_mu (to be determined)
        # the stress values are projected and plotted
        # or better do separate script?

        # ### Quadrature space for principal stress output
        # qs = basix.ufl.quadrature_element(
        #     basix_celltype,
        #     value_shape=(),  # type: ignore
        #     scheme='default',
        #     degree=q_degree,
        # )
        # Q = df.fem.functionspace(submesh, qs)
        # p_stress_fom = df.fem.Function(Q)
        # p_stress_rom = df.fem.Function(Q)

        # ### Lagrange space for stress output
        # W = df.fem.functionspace(submesh, ('P', example.fe_deg))  # output space for stress

        # p_stress_rom.x.array[:] = s_rom
        # p_stress_fom.x.array[:] = s_fom
        # if i_mu == 1:
        #     from parageom.stress_analysis import project
        #
        #     s_error_q = df.fem.Function(p_stress_fom.function_space)
        #     s_error_q.x.array[:] = np.abs(p_stress_fom.x.array - p_stress_rom.x.array)
        #
        #     proj_s_error = df.fem.Function(W, name='serr')
        #     proj_stress_fom = df.fem.Function(W, name='sfom')
        #     proj_stress_rom = df.fem.Function(W, name='srom')
        #
        #     project(s_error_q, proj_s_error)
        #     project(p_stress_fom, proj_stress_fom)
        #     project(p_stress_rom, proj_stress_rom)
        #
        #     with df.io.XDMFFile(W.mesh.comm, 'output/stress_error.xdmf', 'w') as xdmf:
        #         xdmf.write_mesh(W.mesh)
        #         xdmf.write_function(proj_s_error)
        #     with df.io.XDMFFile(W.mesh.comm, 'output/stress_fom.xdmf', 'w') as xdmf:
        #         xdmf.write_mesh(W.mesh)
        #         xdmf.write_function(proj_stress_fom)
        #     with df.io.XDMFFile(W.mesh.comm, 'output/stress_rom.xdmf', 'w') as xdmf:
        #         xdmf.write_mesh(W.mesh)
        #         xdmf.write_function(proj_stress_rom)

    # displacement error (energy norm)
    u_error = ufom_sols - urom_sols
    u_error_norm = u_error.norm(energy_product) / ufom_sols.norm(energy_product)

    # transformation displacement error
    # d_error = dfom_sols - drom_sols
    # d_error_norm = d_error.norm(energy_product) / dfom_sols.norm(energy_product)

    # stress error (Euclidean norm)
    s_error = sfom_sols - srom_sols
    s_error_norm = s_error.norm() / sfom_sols.norm()

    # scale each vector by respective max value of FOM solution
    u_error.scal(1 / ufom_sols.sup_norm())
    s_error.scal(1 / sfom_sols.sup_norm())
    # take the max value over all nodes
    max_nodal_displacement_error = u_error.sup_norm()
    max_nodal_stress_error = s_error.sup_norm()
    assert max_nodal_displacement_error.size == len(validation_set)
    assert max_nodal_stress_error.size == len(validation_set)

    logger.info(f"""Summary
    Validation set size = {len(validation_set)}
    Num modes = {num_modes}
    With EI = {str(args.ei)}

    Displacement
          Relative Error in Energy Norm:
          ---------------------
          min = {np.min(u_error_norm)}
          max = {np.max(u_error_norm)}
          avg = {np.average(u_error_norm)}
          Worst mu = {np.argmax(u_error_norm)}

          Max Nodal Relative Error:
          ----------------
          min = {np.min(max_nodal_displacement_error)}
          max = {np.max(max_nodal_displacement_error)}
          avg = {np.average(max_nodal_displacement_error)}

    Stress
          Relative Error in Euclidean Norm:
          ---------------------
          min = {np.min(s_error_norm)}
          max = {np.max(s_error_norm)}
          avg = {np.average(s_error_norm)}
          Worst mu = {np.argmax(s_error_norm)}

          Max Nodal Relative Error:
          ----------------
          min = {np.min(max_nodal_stress_error)}
          max = {np.max(max_nodal_stress_error)}
          avg = {np.average(max_nodal_stress_error)}
    """)

    # ### Write targets
    def outdata(array, norm):
        r = {}
        r[f'{norm}_max'] = np.max(array)
        r[f'{norm}_min'] = np.min(array)
        r[f'{norm}_avg'] = np.average(array)
        return r

    # u - min,avg,max over validation set for relative error in energy norm
    output_u = example.rom_error(
        args.method, args.nreal, example.rom_validation.fields[0], num_modes, args.ei
    ).as_posix()
    np.savez(output_u, **outdata(u_error_norm, 'energy'), **outdata(max_nodal_displacement_error, 'max'))

    # s - min,avg,max over validation set for max relative nodal error
    output_s = example.rom_error(
        args.method, args.nreal, example.rom_validation.fields[1], num_modes, args.ei
    ).as_posix()
    np.savez(output_s, **outdata(s_error_norm, 'euclidean'), **outdata(max_nodal_stress_error, 'max'))

    if args.condition:
        assert kappa is not None
        output_k = example.rom_condition(args.nreal, args.num_modes, method=args.method, ei=args.ei)
        np.save(output_k, kappa)


def build_fom(example, ω=0.5):
    from parageom.auxiliary_problem import discretize_auxiliary_problem
    from parageom.fom import discretize_fom

    coarse_grid_path = example.coarse_grid
    coarse_grid = StructuredQuadGrid(*read_mesh(coarse_grid_path, MPI.COMM_WORLD, kwargs={'gdim': example.gdim}))
    parent_domain_path = example.fine_grid
    omega_gl = RectangularDomain(*read_mesh(parent_domain_path, MPI.COMM_WORLD, kwargs={'gdim': example.gdim}))
    interface_tags = [i for i in range(15, 25)]
    auxiliary_problem = discretize_auxiliary_problem(
        example,
        omega_gl,
        interface_tags,
        Parameters({example.parameter_name: example.nx * example.ny}),
        coarse_grid=coarse_grid,
    )
    trafo_disp = df.fem.Function(auxiliary_problem.problem.V, name='d_μ_fom')
    fom, parageom = discretize_fom(example, auxiliary_problem, trafo_disp, ω=ω)
    return fom, parageom


def build_rom(example, dofmap, params, num_modes, ω=0.5, nreal=0, method='hapod', use_ei=False):
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    from parageom.ei import interpolate_subdomain_operator
    from parageom.fom import discretize_subdomain_operators
    from parageom.locmor import EISubdomainOperatorWrapper, assemble_gfem_system_with_ei

    # local high fidelity operators
    operator_local, rhs_local, theta_vol, auxmodel = discretize_subdomain_operators(example)

    # ### Reduced bases
    num_coarse_grid_cells = dofmap.grid.num_cells
    local_bases = []
    max_dofs_per_vert = []
    for cell in range(num_coarse_grid_cells):
        local_bases.append(np.load(example.local_basis_npy(nreal, cell, method=method)))
        max_dofs_per_vert.append(np.load(example.local_basis_dofs_per_vert(nreal, cell, method=method)))

    # ### Maximum number of modes per vertex
    max_dofs_per_vert = np.array(max_dofs_per_vert)
    assert max_dofs_per_vert.shape == (num_coarse_grid_cells, 4)
    bases_length = [len(rb) for rb in local_bases]
    assert np.allclose(np.array(bases_length), np.sum(max_dofs_per_vert, axis=1))

    # ### DofMap Dof Distribution
    dofs_per_vert = max_dofs_per_vert.copy()
    dofs_per_vert[max_dofs_per_vert > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vert)

    # ### Volume output
    def compute_volume(mu):
        vol = 0.0
        for mu_i in mu.to_numpy():
            loc_mu = theta_vol.parameters.parse([mu_i])
            vol += theta_vol.evaluate(loc_mu)
        return vol

    theta_vol_gl = GenericParameterFunctional(compute_volume, params)
    initial_mu = params.parse([0.1 for _ in range(num_coarse_grid_cells)])
    vol_ref = theta_vol_gl.evaluate(initial_mu)
    vol_va = NumpyVectorSpace(1).ones(1)
    vol_va.scal((1.0 - ω) / vol_ref)

    # V =  one_op(vol_va) * theta_vol_gl
    # initial_mu --> U_ref
    # compliance ( Uref, rhs)
    # ω
    # J = V + ω compliance
    # output_args = {
    #         "vol_va": vol_va,
    #         "theta_vol_gl": theta_vol_gl,
    #         "initial_mu": initial_mu,
    #         "omega": ω
    #         }

    # EI
    wrapped_op = None
    if use_ei:
        mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(
            example,
            operator_local,
            design='uniform',
            ntrain=501,
            modes=None,
            atol=0.0,
            rtol=example.mdeim_rtol,
            l2_err=example.mdeim_l2err,
            method='method_of_snapshots',
        )
        m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
        restricted_op, _ = operator_local.restricted(m_dofs, padding=1e-8)
        wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix, magic_dofs, m_inv)
        # convert `rhs_local` to NumPy
        vector = rhs_local.as_range_array().to_numpy()  # type: ignore
        rhs_va = mops[0].range.from_numpy(vector)
        rhs_local = VectorOperator(rhs_va)

        # Assembley of global operators
        operator, rhs, selected_modes = assemble_gfem_system_with_ei(
            dofmap, wrapped_op, rhs_local, local_bases, dofs_per_vert, max_dofs_per_vert, params
        )

        # Compliance
        U_ref = operator.apply_inverse(rhs.as_range_array(), mu=initial_mu)
        compl_ref = rhs.as_range_array().inner(U_ref).item()
        scaled_fext = rhs.as_range_array()
        scaled_fext.scal(1.0 / compl_ref)
        compliance = VectorFunctional(scaled_fext, product=None, name='compliance')

        # Output definition
        one_op = ConstantOperator(vol_va, source=operator.source)
        output = LincombOperator([one_op, compliance], [theta_vol_gl, ω])

        rom = StationaryModel(operator, rhs, output_functional=output, name='ROM_with_ei')
        return rom, selected_modes, auxmodel
    else:
        return {
            'dofmap': dofmap,
            'operator_local': operator_local,
            'rhs_local': rhs_local,
            'mu': None,
            'local_bases': local_bases,
            'dofs_per_vert': dofs_per_vert,
            'max_dofs_per_vert': max_dofs_per_vert,
        }, auxmodel


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Compute ROM error over validation set.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('nreal', type=int, help='The nreal-th realization of the local bases.')
    parser.add_argument('method', type=str, help='The method used to construct local bases.', choices=('hapod', 'hrrf'))
    parser.add_argument('num_params', type=int, help='Size of the validation set.')
    parser.add_argument('num_modes', type=int, help='Number of modes per vertex of ROM.')
    parser.add_argument('--ei', action='store_true', help='Use EI.')
    parser.add_argument('--omega', type=float, help='Weighting for output functional.', default=0.5)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('--condition', action='store_true', help='Compute condition numbers of ROM operator.')
    # TODO add arg --write-stress-output
    args = parser.parse_args(sys.argv[1:])
    main(args)
