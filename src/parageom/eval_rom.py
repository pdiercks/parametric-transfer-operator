from mpi4py import MPI
import ufl
import basix
import dolfinx as df
import numpy as np
from pymor.tools.random import new_rng
from pymor.models.basic import StationaryModel
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid


def main():
    """Eval ROM for same parameter value twice.
    Call reconstruct in-between.
    See if solutions are equal.
    """
    from .tasks import example
    from .dofmap_gfem import GFEMDofMap
    from .locmor import reconstruct, assemble_gfem_system
    # from .stress_analysis import project

    # ### FOM
    fom, parageom_fom = build_fom(example)
    V = fom.solution_space.V

    # ### Global function for displacment
    d_rom = df.fem.Function(V, name="urom")
    d_fom = df.fem.Function(V, name="ufom")
    other_d_rom = df.fem.Function(V)

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
        # FIXME
        # how to avoid hardcoding reshape?
        values = f.reshape(cells.size, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx+fyy) / 2 - np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        fmax = (fxx+fyy) / 2 + np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        return fmin, fmax

    # ### Quadrature space for principal stress
    qs = basix.ufl.quadrature_element(basix_celltype, value_shape=(2,), # type: ignore
                                      scheme="default", degree=q_degree)
    Q = df.fem.functionspace(V.mesh, qs)
    p_stress_fom = df.fem.Function(Q)
    p_stress_rom = df.fem.Function(Q)

    # ### Lagrange space for stress output
    W = df.fem.functionspace(V.mesh, ("P", 2, (2,))) # output space: linear Lagrange elements
    proj_stress_fom = df.fem.Function(W)
    proj_stress_rom = df.fem.Function(W)

    # ### Function for displacement on unit cell (for reconstruction)
    unit_cell_domain = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": 2})[0]
    V_i = df.fem.functionspace(unit_cell_domain, ("P", 2, (2, )))
    d_local = df.fem.Function(V_i, name="u_i")

    # ### Build localized ROM
    print("Building ROM ...")
    coarse_grid_path = example.coarse_grid("global")

    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": 2})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)
    dofmap = GFEMDofMap(struct_grid)
    params = example.parameters["global"]
    num_modes = 50

    rom = None
    use_ei = False
    if use_ei:
        rom, modes = build_rom(example, dofmap, params, num_modes, use_ei=use_ei)
        rom_data = {}
    else:
        rom_data = build_rom(example, dofmap, params, num_modes, use_ei=use_ei)

    P = params.space(example.mu_range)
    with new_rng(example.validation_set_seed):
        validation_set = P.sample_randomly(100)

    # mu = validation_set[79]
    # u = rom.solve(mu)
    # print(u)
    # reconstruct(u.to_numpy(), dofmap, modes, d_local, d_rom) # type: ignore
    # uu = rom.solve(mu)
    # print(uu)
    # assert np.allclose(u.to_numpy(), uu.to_numpy()) # type: ignore

    fom_sols = fom.solution_space.empty()
    rom_sols = fom.solution_space.empty()
    other_rom_sols = fom.solution_space.empty()

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

    validation_set = validation_set[61:71]

    for mu in validation_set:
        U_fom = fom.solve(mu)
        fom_sols.append(U_fom)
        d_fom.x.array[:] = U_fom.to_numpy().flatten() # type: ignore

        if use_ei:
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
            rom = StationaryModel(operator, rhs, name="locROM")
            urb = rom.solve(mu)
        reconstruct(urb.to_numpy(), dofmap, modes, d_local, d_rom) # type: ignore
        U_rom = fom.solution_space.make_array([d_rom.x.petsc_vec.copy()]) # type: ignore
        rom_sols.append(U_rom)

        # try different solver for rom
        print("Trying different solver (dense)")
        A = rom.operator.assemble(mu).matrix.todense()
        b = rom.rhs.as_range_array(mu).to_numpy().flatten()
        x = np.linalg.solve(A, b)

        reconstruct(x.reshape(urb.to_numpy().shape), dofmap, modes, d_local, other_d_rom)
        other_rom_sols.append(fom.solution_space.make_array([other_d_rom.x.petsc_vec.copy()])) # type: ignore
        stress_expr_fom.eval(V.mesh, entities=cells, values=stress_fom.x.array.reshape(cells.size, -1))
        s_fom = compute_principal_components(stress_fom.x.array.reshape(cells.size, -1))

        stress_expr_rom.eval(V.mesh, entities=cells, values=stress_rom.x.array.reshape(cells.size, -1))
        s_rom = compute_principal_components(stress_rom.x.array.reshape(cells.size, -1))

        first_principal = compute_stress_error_norms(s_fom[1], s_rom[1])
        max_rel_err_stress.append(first_principal["max_rel_err"])
        

    u_errors = fom_sols - rom_sols
    other_u_errors = fom_sols - other_rom_sols
    errn = u_errors.norm(energy_product) / fom_sols.norm(energy_product)
    other_errn = other_u_errors.norm(energy_product) / fom_sols.norm(energy_product)
    breakpoint()
    # compare errn and err_norms

    # np.sqrt(product.pairwise_apply2(U, U))

    print(f"Num modes = {num_modes}\n")

    print(f"""Displacement
          min err = {np.min(errn)}
          max err = {np.max(errn)}
          avg err = {np.average(errn)}
          """)

    print(f"""Stress
          min (max) rel err = {np.min(max_rel_err_stress)}
          max (max) rel err = {np.max(max_rel_err_stress)}
          avg (max) rel err = {np.average(max_rel_err_stress)}
          """)

    print(f"Worst mu (displacement): {np.argmax(errn)}")
    print(f"Worst mu (stress): {np.argmax(max_rel_err_stress)}")

    breakpoint()

    # for loop over validation set, but const. num_modes
    # expect that error in U and S are of the same order --> no outliers


def build_fom(example, ω=0.5):
    from .fom import discretize_fom
    from .auxiliary_problem import discretize_auxiliary_problem

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


def build_rom(example, dofmap, params, num_modes, nreal=0, method="hapod", distribution="normal", use_ei=False):
    from pymor.operators.constructions import VectorOperator
    from pymor.models.basic import StationaryModel
    from .fom import discretize_subdomain_operators
    from .locmor import assemble_gfem_system_with_ei, EISubdomainOperatorWrapper
    from .ei import interpolate_subdomain_operator

    # local high fidelity operators
    operator_local, rhs_local = discretize_subdomain_operators(example)

    # ### Reduced bases
    # 0: left, 1: transition, 2: inner, 3: transition, 4: right
    archetypes = []
    for cell in range(5):
        archetypes.append(np.load(example.local_basis_npy(nreal, method, distribution, cell)))

    local_bases = []
    local_bases.append(archetypes[0].copy())
    local_bases.append(archetypes[1].copy())
    for _ in range(6):
        local_bases.append(archetypes[2].copy())
    local_bases.append(archetypes[3].copy())
    local_bases.append(archetypes[4].copy())


    # ### Maximum number of modes per vertex
    num_coarse_grid_cells = dofmap.grid.num_cells
    max_dofs_per_vert = np.load(example.local_basis_dofs_per_vert(nreal, method, distribution))
    # raise to number of cells in the coarse grid
    repetitions = [1, 1, num_coarse_grid_cells - len(archetypes) + 1, 1, 1]
    assert np.isclose(np.sum(repetitions), num_coarse_grid_cells)
    max_dofs_per_vert = np.repeat(max_dofs_per_vert, repetitions, axis=0)
    assert max_dofs_per_vert.shape == (num_coarse_grid_cells, 4)
    bases_length = [len(rb) for rb in local_bases]
    assert np.allclose(np.array(bases_length), np.sum(max_dofs_per_vert, axis=1))

    # ### DofMap Dof Distribution
    dofs_per_vert = max_dofs_per_vert.copy()
    dofs_per_vert[max_dofs_per_vert > num_modes] = num_modes
    dofmap.distribute_dofs(dofs_per_vert)

    # TODO integrate output again
    # define second (global) auxiliary problem to be able to compute volume
    # otherwise define volume_operator on unit cell level and take the sum over all cells

    # However, ParaGeomLinEla is also required to compute stress if I want to keep
    # the FOM and ROM objects separate.
    # from .auxiliary_problem import discretize_auxiliary_problem
    #
    # coarse_grid_path = example.coarse_grid("global").as_posix()
    # parent_domain_path = example.parent_domain("global").as_posix()
    # interface_tags = [i for i in range(15, 25)]
    # auxiliary_problem = discretize_auxiliary_problem(
    #     example,
    #     parent_domain_path,
    #     interface_tags,
    #     example.parameters["global"],
    #     coarse_grid=coarse_grid_path,
    # )
    # trafo_disp = df.fem.Function(auxiliary_problem.problem.V, name="d_μ_rom")
    #
    # omega = auxiliary_problem.problem.domain
    # matparam = {"gdim": omega.gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    # parageom = ParaGeomLinEla(omega, auxiliary_problem.problem.V, trafo_disp, matparam)

    # ### Issue with Stress computation
    # this would require global `trafo_displacement` which does not exist for ROM
    # in case of the ROM we only have a local (unit cell) auxiliary problem

    # Conlusion
    # compute output via local operators
    # use global ParaGeomLinEla for stress computation

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
        rom = StationaryModel(operator, rhs, name="ROM_with_ei")
        return rom, selected_modes
    else:
        # operator, rhs, selected_modes = assemble_gfem_system(
        #     dofmap,
        #     operator_local,
        #     rhs_local,
        #     mu,
        #     local_bases,
        #     dofs_per_vert,
        #     max_dofs_per_vert,
        # )
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
    main()
