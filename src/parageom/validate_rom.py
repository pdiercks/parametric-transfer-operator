import pathlib
from time import perf_counter

from mpi4py import MPI
import ufl
import basix
import dolfinx as df
import numpy as np
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid

from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.tools.random import new_rng
from pymor.models.basic import StationaryModel
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import ConstantOperator, LincombOperator, VectorFunctional


def main(args):
    from parageom.tasks import example
    from parageom.dofmap_gfem import GFEMDofMap
    from parageom.locmor import reconstruct, assemble_gfem_system

    stem = pathlib.Path(__file__).stem
    logfilename = example.log_validate_rom(args.nreal, args.num_modes, method=args.method, ei=args.ei).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    if args.debug:
        loglevel = 10 # debug
    else:
        loglevel = 20 # info
    logger = getLogger(stem, level=loglevel)

    # ### FOM
    fom, parageom_fom = build_fom(example)
    V = fom.solution_space.V

    # ### Global function for displacment
    d_rom = df.fem.Function(V, name="urom")
    d_fom = df.fem.Function(V, name="ufom")

    # ### Quadrature space for stress
    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    q_degree = 2
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    qve = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree) # type: ignore
    QV = df.fem.functionspace(V.mesh, qve)

    stress_fom = df.fem.Function(QV)
    stress_rom = df.fem.Function(QV)

    # ### UFL representation and Expression of stress
    suf = parageom_fom.weighted_stress(d_fom)
    stress_ufl_fom_vector = ufl.as_vector([suf[0, 0], suf[1, 1], suf[2, 2], suf[0, 1]])
    stress_expr_fom = df.fem.Expression(stress_ufl_fom_vector, q_points)

    sur = parageom_fom.weighted_stress(d_rom)
    stress_ufl_rom_vector = ufl.as_vector([sur[0, 0], sur[1, 1], sur[2, 2], sur[0, 1]])
    stress_expr_rom = df.fem.Expression(stress_ufl_rom_vector, q_points)

    tdim = V.mesh.topology.dim
    map_c = V.mesh.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    def compute_principal_components(f):
        values = f.reshape(cells.size, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx+fyy) / 2 - np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        fmax = (fxx+fyy) / 2 + np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        return fmin, fmax

    # TODO: add stress plots in other postproc script?

    # ### Quadrature space for principal stress
    # qs = basix.ufl.quadrature_element(basix_celltype, value_shape=(2,), # type: ignore
    #                                   scheme="default", degree=q_degree)
    # Q = df.fem.functionspace(V.mesh, qs)
    # p_stress_fom = df.fem.Function(Q)
    # p_stress_rom = df.fem.Function(Q)

    # ### Lagrange space for stress output
    # W = df.fem.functionspace(V.mesh, ("P", 2, (2,))) # output space: linear Lagrange elements
    # proj_stress_fom = df.fem.Function(W)
    # proj_stress_rom = df.fem.Function(W)

    # ### Function for displacement on unit cell (for reconstruction)
    unit_cell_domain = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": 2})[0]
    V_i = df.fem.functionspace(unit_cell_domain, ("P", 2, (2, )))
    d_local = df.fem.Function(V_i, name="u_i")

    # ### Build localized ROM
    coarse_grid_path = example.coarse_grid("global")
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": 2})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)
    dofmap = GFEMDofMap(struct_grid)
    params = example.parameters["global"]
    num_modes = args.num_modes

    rom = None
    if args.ei:
        logger.info("Building ROM with EI ...")
        tic = perf_counter()
        rom, modes = build_rom(example, dofmap, params, num_modes, ω=args.omega, nreal=args.nreal, method=args.method, distribution="normal", use_ei=args.ei)
        logger.info(f"Took {perf_counter()-tic} to build ROM.")
        rom_data = {}
    else:
        logger.info("Building ROM without EI ...")
        # here time is not interesting as the assembly has to be carried out
        # every time model is evaluated for new `mu`
        rom_data = build_rom(example, dofmap, params, num_modes, ω=args.omega, nreal=args.nreal, method=args.method, distribution="normal", use_ei=args.ei)

    P = params.space(example.mu_range)
    with new_rng(example.validation_set_seed):
        validation_set = P.sample_randomly(args.num_params)

    fom_sols = fom.solution_space.empty()
    rom_sols = fom.solution_space.empty()

    def compute_stress_error_norms(fom, rom):
        fom_norm = np.linalg.norm(fom)
        abs_err = np.abs(fom - rom)
        rel_err = abs_err / fom_norm
        max_rel_err = np.max(rel_err) # pointwise
        rel_err_norm = np.linalg.norm(abs_err) / fom_norm # global
        return {
                "max_rel_err": max_rel_err,
                "rel_err_norm": rel_err_norm,
                }

    max_rel_err_stress = []
    energy_product = fom.products['energy']

    for mu in validation_set:
        U_fom = fom.solve(mu)
        fom_sols.append(U_fom)
        d_fom.x.array[:] = U_fom.to_numpy().flatten() # type: ignore

        if args.ei:
            assert rom is not None
            urb = rom.solve(mu)
        else:
            operator, rhs, modes = assemble_gfem_system(
                rom_data["dofmap"],
                rom_data["operator_local"],
                rom_data["rhs_local"],
                mu,
                rom_data["local_bases"],
                rom_data["dofs_per_vert"],
                rom_data["max_dofs_per_vert"],
            )
            rom = StationaryModel(operator, rhs, output_functional=None, name="ROM")
            urb = rom.solve(mu)
        reconstruct(urb.to_numpy(), dofmap, modes, d_local, d_rom) # type: ignore
        U_rom = fom.solution_space.make_array([d_rom.x.petsc_vec.copy()]) # type: ignore
        rom_sols.append(U_rom)

        stress_expr_rom.eval(V.mesh, entities=cells, values=stress_rom.x.array.reshape(cells.size, -1))
        s_rom = compute_principal_components(stress_rom.x.array.reshape(cells.size, -1))

        stress_expr_fom.eval(V.mesh, entities=cells, values=stress_fom.x.array.reshape(cells.size, -1))
        s_fom = compute_principal_components(stress_fom.x.array.reshape(cells.size, -1))

        first_principal = compute_stress_error_norms(s_fom[1], s_rom[1])
        max_rel_err_stress.append(first_principal["max_rel_err"])
        

    # displacement error in energy norm
    u_errors = fom_sols - rom_sols
    errn = u_errors.norm(energy_product) / fom_sols.norm(energy_product)

    # displacement error per node
    u_errors.scal(1 / fom_sols.amax()[1])
    nodal_uerr = u_errors.amax()[1]

    logger.info(f"""Summary
    Validation set size = {len(validation_set)}
    Num modes = {num_modes}
    With EI = {str(args.ei)}

    Displacement
          Relative Error in Energy Norm:
          ---------------------
          min = {np.min(errn)}
          max = {np.max(errn)}
          avg = {np.average(errn)}
          Worst mu = {np.argmax(errn)}

          Max Nodal Relative Error:
          ----------------
          min = {np.min(nodal_uerr)}
          max = {np.max(nodal_uerr)}
          avg = {np.average(nodal_uerr)}

    Stress
          min (max) rel err = {np.min(max_rel_err_stress)}
          max (max) rel err = {np.max(max_rel_err_stress)}
          avg (max) rel err = {np.average(max_rel_err_stress)}
          Worst mu = {np.argmax(max_rel_err_stress)}
    """)

    # ### Write targets
    output_u = example.rom_error_u(args.nreal, num_modes, method=args.method, ei=args.ei).as_posix()
    np.savez(output_u, relerr=errn, nodal_err=nodal_uerr)

    output_s = example.rom_error_s(args.nreal, num_modes, method=args.method, ei=args.ei).as_posix()
    np.savez(output_s, relerr=max_rel_err_stress)


def build_fom(example, ω=0.5):
    from parageom.fom import discretize_fom
    from parageom.auxiliary_problem import discretize_auxiliary_problem

    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    interface_tags = [i for i in range(15, 25)]
    auxiliary_problem = discretize_auxiliary_problem(
        example,
        parent_domain_path,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    trafo_disp = df.fem.Function(auxiliary_problem.problem.V, name="d_μ_fom")
    fom, parageom = discretize_fom(example, auxiliary_problem, trafo_disp, ω=ω)
    return fom, parageom


def build_rom(example, dofmap, params, num_modes, ω=0.5, nreal=0, method="hapod", distribution="normal", use_ei=False):
    from pymor.operators.constructions import VectorOperator
    from pymor.models.basic import StationaryModel
    from parageom.fom import discretize_subdomain_operators
    from parageom.locmor import assemble_gfem_system_with_ei, EISubdomainOperatorWrapper
    from parageom.ei import interpolate_subdomain_operator

    # local high fidelity operators
    operator_local, rhs_local, theta_vol = discretize_subdomain_operators(example)

    # ### Reduced bases
    num_coarse_grid_cells = dofmap.grid.num_cells
    local_bases = []
    max_dofs_per_vert = []
    for cell in range(num_coarse_grid_cells):
        local_bases.append(np.load(example.local_basis_npy(nreal, cell, method=method, distr=distribution)))
        max_dofs_per_vert.append(np.load(example.local_basis_dofs_per_vert(nreal, cell, method=method, distr=distribution)))

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
    vol_va.scal( (1. - ω) / vol_ref )

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
        mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator_local, design="uniform", ntrain=501, modes=None, atol=0., rtol=example.mdeim_rtol, method="method_of_snapshots")
        m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
        restricted_op, _ = operator_local.restricted(m_dofs, padding=1e-8)
        wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix, magic_dofs, m_inv)
        # convert `rhs_local` to NumPy
        vector = rhs_local.as_range_array().to_numpy() # type: ignore
        rhs_va = mops[0].range.from_numpy(vector)
        rhs_local = VectorOperator(rhs_va)

        # Assembley of global operators
        operator, rhs, selected_modes = assemble_gfem_system_with_ei(
                dofmap, wrapped_op, rhs_local, local_bases, dofs_per_vert, max_dofs_per_vert,
                params)

        # Compliance
        U_ref = operator.apply_inverse(rhs.as_range_array(), mu=initial_mu)
        compl_ref = rhs.as_range_array().inner(U_ref).item()
        scaled_fext = rhs.as_range_array()
        scaled_fext.scal(1. / compl_ref)
        compliance = VectorFunctional(scaled_fext, product=None, name="compliance")

        # Output definition
        one_op = ConstantOperator(vol_va, source=operator.source)
        output = LincombOperator([one_op, compliance], [theta_vol_gl, ω])

        rom = StationaryModel(operator, rhs, output_functional=output, name="ROM_with_ei")
        return rom, selected_modes
    else:
        return {
                "dofmap": dofmap,
                "operator_local": operator_local,
                "rhs_local": rhs_local,
                "mu": None,
                "local_bases": local_bases,
                "dofs_per_vert": dofs_per_vert,
                "max_dofs_per_vert": max_dofs_per_vert,
                }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute ROM error over validation set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nreal", type=int, help="The nreal-th realization of the local bases.")
    parser.add_argument("method", type=str, help="The method used to construct local bases.",
                        choices=("hapod", "heuristic"))
    parser.add_argument("num_params", type=int, help="Size of the validation set.")
    parser.add_argument("num_modes", type=int, help="Number of modes per vertex of ROM.")
    parser.add_argument("--ei", action="store_true", help="Use EI.")
    parser.add_argument("--omega", type=float, help="Weighting for output functional.", default=0.5)
    parser.add_argument("--debug", action='store_true', help="Run in debug mode.")
    # TODO add arg --write-stress-output
    args = parser.parse_args(sys.argv[1:])
    main(args)
