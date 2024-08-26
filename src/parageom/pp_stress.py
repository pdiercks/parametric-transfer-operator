"""post-processing of stress"""

import dolfinx as df
import basix

import numpy as np
from pymor.core.pickle import load
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.operators.constructions import VectorOperator, LincombOperator, VectorFunctional, ConstantOperator
from pymor.parameters.functionals import GenericParameterFunctional
from pymor.models.basic import StationaryModel


def main(args):
    """compute principal stress using FOM and ROM"""
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom, ParaGeomLinEla

    # ### Build FOM
    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    interface_tags = [
        i for i in range(15, 25)
    ]
    auxiliary_problem = discretize_auxiliary_problem(
        example,
        parent_domain_path,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    V = auxiliary_problem.problem.V
    d_trafo = df.fem.Function(V, name="d_trafo")
    fom = discretize_fom(example, auxiliary_problem, d_trafo, ω=args.omega)

    gdim = fom.solution_space.V.mesh.geometry.dim
    matparam = {"gdim": gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(auxiliary_problem.problem.domain, V, d_trafo, matparam) # type: ignore

    # ### Build localized ROM
    rom, rec_data = build_localized_rom(args, example, auxiliary_problem, d_trafo, fom.parameters, ω=args.omega)

    # ### Global function for stress
    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    q_degree = 2
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree) # type: ignore
    QV = df.fem.functionspace(V.mesh, QVe)
    stress = df.fem.Function(QV, name="Cauchy")

    # ### Global function for displacment
    displacement = df.fem.Function(V)

    # ### Get optimal solution μ*
    fom_data = example.fom_minimization_data
    with fom_data.open("rb") as fh:
        data = load(fh)
    mu = fom.parameters.parse(data["mu_min"])

    # targets
    xdmf_files = example.pp_stress(args.name)

    s1_fom, s2_fom, s_fom = compute_principal_stress(fom, mu, displacement, stress, parageom, rec_data=None, xdmf_filename=xdmf_files["fom"].as_posix())
    displacement.x.array[:] = 0. # type: ignore
    stress.x.array[:] = 0. # type: ignore
    s1_rom, s2_rom, s_rom = compute_principal_stress(rom, mu, displacement, stress, parageom, rec_data=rec_data, xdmf_filename=xdmf_files["rom"].as_posix())

    # stress error in euclidean norm
    def compute_norms(fom, rom):
        err = np.abs(fom.flatten() - rom.flatten())
        en = np.linalg.norm(err)
        fom_norm = np.linalg.norm(fom.flatten())
        return en, fom_norm

    en_1, fn_1 = compute_norms(s1_fom, s1_rom)
    en_2, fn_2 = compute_norms(s2_fom, s2_rom)
    print(f"""Relative error in Euclidean norm:
          s1: {en_1 / fn_1}
          s2: {en_2 / fn_2}
          """)

    Q = s_fom.function_space # type: ignore
    error = df.fem.Function(Q, name="e")
    sx_max = np.abs(np.amin(s_fom.x.array[::2])) # type: ignore ; compression
    sy_max = np.abs(np.amax(s_fom.x.array[1::2])) # type: ignore ; tension
    error.x.array[:] = np.abs(s_fom.x.array[:] - s_rom.x.array[:]) # type: ignore
    error.x.array[::2] /= sx_max # type: ignore
    error.x.array[1::2] /= sy_max # type: ignore

    with df.io.XDMFFile(Q.mesh.comm, xdmf_files["err"].as_posix(), "w") as xdmf: # type: ignore
        xdmf.write_mesh(Q.mesh)
        xdmf.write_function(error) # type: ignore


def compute_principal_stress(model, mu, u, stress, parageom, rec_data=None, xdmf_filename=None):
    from .stress_analysis import principal_stress_2d, project

    V = u.function_space
    domain = V.mesh
    tdim = domain.topology.dim
    map_c = domain.topology.index_map(tdim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    if model.name.startswith("locROM"):
        assert rec_data is not None
        dofmap, local_bases, u_local = rec_data
        from .locmor import reconstruct

        def update_displacement_solution(mu):
            U = model.solve(mu)
            reconstruct(U.to_numpy(), dofmap, local_bases, u_local, u)
    else:
        assert rec_data is None

        def update_displacement_solution(mu):
            U = model.solve(mu)
            u.x.array[:] = U.to_numpy().flatten()

    update_displacement_solution(mu)
    q_degree = 2
    s1, s2 = principal_stress_2d(u, parageom, q_degree=q_degree, cells=cells, values=stress.x.array.reshape(cells.size, -1))

    # quadrature space and function space for stress output
    basix_celltype = getattr(basix.CellType, V.mesh.topology.cell_type.name)
    qs = basix.ufl.quadrature_element(basix_celltype, value_shape=(2,), scheme="default", degree=q_degree)
    Q = df.fem.functionspace(V.mesh, qs)
    sigma_q = df.fem.Function(Q, name="sp")

    W = df.fem.functionspace(V.mesh, ("P", 2, (2,))) # output space: linear Lagrange elements
    sigma_p = df.fem.Function(W, name="sp")

    sigma_q.x.array[::2] = s1.flatten()
    sigma_q.x.array[1::2] = s2.flatten()
    project(sigma_q, sigma_p)

    if xdmf_filename is not None:
        with df.io.XDMFFile(W.mesh.comm, xdmf_filename, "w") as xdmf:
            xdmf.write_mesh(W.mesh)
            xdmf.write_function(sigma_p)

    return s1, s2, sigma_p


def build_localized_rom(cli, example, global_auxp, trafo_disp, parameters, ω=0.5):
    from .fom import discretize_subdomain_operators, ParaGeomLinEla
    from .ei import interpolate_subdomain_operator
    from .locmor import EISubdomainOperatorWrapper, assemble_gfem_system_with_ei
    from .dofmap_gfem import GFEMDofMap

    nreal = 0

    operator_local, rhs_local = discretize_subdomain_operators(example)
    u_local = df.fem.Function(operator_local.source.V)

    mops, interpolation_matrix, idofs, magic_dofs, deim_data = interpolate_subdomain_operator(example, operator_local, design="uniform", ntrain=501, modes=None, atol=0., rtol=1e-12, method="method_of_snapshots")
    m_dofs, m_inv = np.unique(magic_dofs, return_inverse=True)
    restricted_op, _ = operator_local.restricted(m_dofs, padding=1e-8)
    wrapped_op = EISubdomainOperatorWrapper(restricted_op, mops, interpolation_matrix, magic_dofs, m_inv)

    # convert `rhs_local` to NumPy
    vector = rhs_local.as_range_array().to_numpy()
    rhs_va = mops[0].range.from_numpy(vector)
    rhs_local = VectorOperator(rhs_va)

    # ### Coarse grid of the global domain
    coarse_grid = global_auxp.coarse_grid

    # ### DofMap
    dofmap = GFEMDofMap(coarse_grid)

    # ### Reduced bases
    # 0: left, 1: transition, 2: inner, 3: transition, 4: right
    archetypes = []
    for cell in range(5):
        archetypes.append(np.load(example.local_basis_npy(nreal, cli.name, cli.distr, cell)))

    local_bases = []
    local_bases.append(archetypes[0].copy())
    local_bases.append(archetypes[1].copy())
    for _ in range(6):
        local_bases.append(archetypes[2].copy())
    local_bases.append(archetypes[3].copy())
    local_bases.append(archetypes[4].copy())
    bases_length = [len(rb) for rb in local_bases]
    
    # ### Maximum number of modes per vertex
    max_dofs_per_vert = np.load(example.local_basis_dofs_per_vert(nreal, cli.name, cli.distr))
    # raise to number of cells in the coarse grid
    repetitions = [1, 1, coarse_grid.num_cells - len(archetypes) + 1, 1, 1]
    assert np.isclose(np.sum(repetitions), coarse_grid.num_cells)
    max_dofs_per_vert = np.repeat(max_dofs_per_vert, repetitions, axis=0)
    assert max_dofs_per_vert.shape == (coarse_grid.num_cells, 4)
    assert np.allclose(np.array(bases_length), np.sum(max_dofs_per_vert, axis=1))

    Nmax = max_dofs_per_vert.max()
    ΔN = 10
    num_modes_per_vertex = list(range(Nmax // ΔN, Nmax + 1, 3 * (Nmax // ΔN) ))
    nmodes = num_modes_per_vertex[-2] # second to last point in the validation

    dofs_per_vert = max_dofs_per_vert.copy()
    dofs_per_vert[max_dofs_per_vert > nmodes] = nmodes
    dofmap.distribute_dofs(dofs_per_vert)

    operator, rhs, local_bases = assemble_gfem_system_with_ei(
        dofmap, wrapped_op, rhs_local, local_bases, dofs_per_vert, max_dofs_per_vert, parameters)

    # definition of ParaGeom Problem for volume computation
    omega = global_auxp.problem.domain
    matparam = {"gdim": omega.gdim, "E": 1.0, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(omega, global_auxp.problem.V, trafo_disp, matparam)

    def param_setter(mu):
        trafo_disp.x.array[:] = 0.0
        global_auxp.solve(trafo_disp, mu)
        trafo_disp.x.scatter_forward()

    def compute_volume(mu):
        param_setter(mu)

        vcpp = df.fem.form(parageom.form_volume)
        vol = df.fem.assemble_scalar(vcpp) # type: ignore
        return vol

    initial_mu = parameters.parse([0.1 for _ in range(example.nx)])
    vol_ref = compute_volume(initial_mu)
    U_ref = operator.apply_inverse(rhs.as_range_array(), mu=initial_mu)

    volume = GenericParameterFunctional(compute_volume, parameters)
    vol_va = NumpyVectorSpace(1).ones(1)
    vol_va.scal( (1. - ω) / vol_ref)
    one_op = ConstantOperator(vol_va, source=operator.source)

    # compliance
    compl_ref = rhs.as_range_array().inner(U_ref).item()
    scaled_fext = rhs.as_range_array().copy()
    scaled_fext.scal(1 / compl_ref)
    compliance = VectorFunctional(scaled_fext, product=None, name="compliance")

    # output J = (1 - ω) mass + ω compliance
    output = LincombOperator([one_op, compliance], [volume, ω])

    rom = StationaryModel(operator, rhs, output_functional=output, name="locROM_with_ei")
    return rom, (dofmap, local_bases, u_local)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute principal stress using FOM and localized ROM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "distr",
        type=str,
        help="The distribution used for sampling.",
        choices=("normal", "multivariate_normal"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument(
            "--omega",
            type=float,
            help="Weighting factor for output functional.",
            default=0.5
            )
    args = parser.parse_args(sys.argv[1:])
    main(args)
