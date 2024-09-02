"""oversampling version 2"""

import typing
from dataclasses import dataclass
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid, RectangularDomain
from multi.preprocessing import create_meshtags
from multi.boundary import within_range, plane_at, point_at
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.solver import build_nullspace
from multi.projection import orthogonal_part

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.parameters.base import Parameters, ParameterSpace
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.tools.random import new_rng

from scipy.sparse import csr_array


@dataclass
class OversamplingConfig:
    cells_omega: npt.NDArray[np.int32]
    cells_omega_in: npt.NDArray[np.int32]
    kernel: tuple[int]
    gamma_out: typing.Callable
    gamma_d: typing.Optional[typing.Callable] = None
    gamma_n: typing.Optional[typing.Callable] = None


def oversampling_config_factory(k):
    """Creates instance of `OversamplingConfig`"""

    cells_omega = {
            0: np.array([0, 1], dtype=np.int32),
            1: np.array([0, 1, 2], dtype=np.int32),
            2: np.array([0, 1, 2, 3], dtype=np.int32),
            3: np.array([1, 2, 3, 4], dtype=np.int32),
            4: np.array([2, 3, 4, 5], dtype=np.int32),
            5: np.array([3, 4, 5, 6], dtype=np.int32),
            6: np.array([4, 5, 6, 7], dtype=np.int32),
            7: np.array([5, 6, 7, 8], dtype=np.int32),
            8: np.array([6, 7, 8, 9], dtype=np.int32),
            9: np.array([7, 8, 9], dtype=np.int32),
            10: np.array([8, 9], dtype=np.int32),
            }

    cells_omega_in = {
            0: np.array([0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([1, 2], dtype=np.int32),
            3: np.array([2, 3], dtype=np.int32),
            4: np.array([3, 4], dtype=np.int32),
            5: np.array([4, 5], dtype=np.int32),
            6: np.array([5, 6], dtype=np.int32),
            7: np.array([6, 7], dtype=np.int32),
            8: np.array([7, 8], dtype=np.int32),
            9: np.array([8, 9], dtype=np.int32),
            10: np.array([9], dtype=np.int32),
            }

    kernel = {
            0: (1, 2),
            1: (1, 2),
            2: (0, 1, 2),
            3: (0, 1, 2),
            4: (0, 1, 2),
            5: (0, 1, 2),
            6: (0, 1, 2),
            7: (0, 1, 2),
            8: (0, 1, 2),
            9: (0, 2),
            10: (0, 2),
            }
    # required enrichment should be determined from kernel

    # ### Topology
    x_max = 10.0
    y_max = 1.0
    unit_length = 1.0
    support_left = plane_at(0.0, "x")
    support_right = point_at([x_max, 0.0, 0.0])
    neumann_top = within_range([0.0, y_max, 0.0], [unit_length, y_max, 0.0])

    # define left based on smallest cell integer
    left_most_cell = np.amin(cells_omega[k])
    x_left = float(left_most_cell * unit_length)
    right_most_cell = np.amax(cells_omega[k])
    x_right = float((right_most_cell + 1) * unit_length)
    # define right based on largest cell integer
    tol = 1e-4
    left = within_range([x_left, 0.0 + tol, 0.0], [x_left, y_max - tol, 0.0])
    right = within_range([x_right, 0.0 + tol, 0.0], [x_right, y_max - tol, 0.0])
    def gamma_out_inner(x):
        return np.logical_or(left(x), right(x))
    gamma_out = {
            0: right,
            1: right,
            2: right,
            3: gamma_out_inner,
            4: gamma_out_inner,
            5: gamma_out_inner,
            6: gamma_out_inner,
            7: gamma_out_inner,
            8: left,
            9: left,
            10: left,
            }
    if k in (0, 1, 2):
        return OversamplingConfig(cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k], gamma_d=support_left, gamma_n=neumann_top)
    elif k in (8, 9, 10):
        return OversamplingConfig(cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k], gamma_d=support_right, gamma_n=None)
    else:
        return OversamplingConfig(cells_omega[k], cells_omega_in[k], kernel[k], gamma_out[k])


def main(args):
    from .tasks import example
    from .lhs import sample_lhs
    from .preprocessing import create_fine_scale_grid_v2, create_structured_coarse_grid_v2
    from .matrix_based_operator import FenicsxMatrixBasedOperator, BCGeom, BCTopo, _create_dirichlet_bcs
    from .auxiliary_problem import GlobalAuxiliaryProblem
    from .fom import ParaGeomLinEla
    from .locmor import DirichletLift, ParametricTransferProblem
    from .hapod import adaptive_rrf_normal

    # ### Coarse grid partition
    coarse_grid_path = example.coarse_grid("global")
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": example.gdim})[0]
    struct_grid_gl = StructuredQuadGrid(coarse_domain)

    osp_config = oversampling_config_factory(args.k)
    cells_omega = osp_config.cells_omega
    cells_omega_in = osp_config.cells_omega_in

    # create coarse grid partition of oversampling domain
    outstream = example.path_omega_coarse(args.k)
    create_structured_coarse_grid_v2(example, struct_grid_gl, cells_omega, outstream.as_posix())
    coarse_omega = read_mesh(outstream, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})[0]
    struct_grid_omega = StructuredQuadGrid(coarse_omega)
    assert struct_grid_omega.num_cells == cells_omega.size

    # create fine grid partition of oversampling domain
    output = example.path_omega(args.k)
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega, output.as_posix())
    omega, omega_ct, omega_ft = read_mesh(output, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})
    omega = RectangularDomain(omega, cell_tags=omega_ct, facet_tags=omega_ft)
    # create facets
    # facet tags for void interfaces start from 15 (see create_fine_scale_grid_v2)
    # i.e. 15 ... 24 for max number of cells

    facet_tag_definitions = {}
    for tag, key in zip([int(11), int(12), int(13)], ["bottom", "left", "right"]):
        facet_tag_definitions[key] = (tag, omega.str_to_marker(key))

    # add tags for neumann boundary
    if osp_config.gamma_n is not None:
        top_tag = int(194)
        top_locator = osp_config.gamma_n
        facet_tag_definitions["top"] = (top_tag, top_locator)

    # update already existing facet tags
    # this will add tags for "top" boundary
    omega.facet_tags = create_meshtags(omega.grid, omega.tdim-1, facet_tag_definitions, tags=omega.facet_tags)[0]

    assert omega.facet_tags.find(11).size == example.num_intervals * cells_omega.size  # bottom
    assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
    assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
    for itag in range(15, 15 + cells_omega.size):
        assert omega.facet_tags.find(itag).size == example.num_intervals * 4  # void

    # create fine grid partition of target subdomain
    output = example.path_omega_in(args.k)
    create_fine_scale_grid_v2(example, struct_grid_gl, cells_omega_in, output.as_posix())
    omega_in, omega_in_ct, omega_in_ft = read_mesh(output, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})
    omega_in = RectangularDomain(omega_in, cell_tags=omega_in_ct, facet_tags=omega_in_ft)

    # create necessary connectivities
    omega.grid.topology.create_connectivity(0, 2)
    omega_in.grid.topology.create_connectivity(0, 2)

    # ### Function Spaces
    V = df.fem.functionspace(omega.grid, ("P", example.geom_deg, (example.gdim,)))
    V_in = df.fem.functionspace(omega_in.grid, V.ufl_element())
    source_V_in = FenicsxVectorSpace(V_in)

    # ### Dirichlet BCs
    # have to be defined twice (operator & range product)

    bcs_op = [] # BCs for lhs operator of transfer problem, space V
    bcs_range_product = [] # BCs for range product operator, space V_in

    zero = df.default_scalar_type(0.0)
    fix_u = df.fem.Constant(V.mesh, (zero,) * example.gdim)
    bc_gamma_out = BCGeom(fix_u, osp_config.gamma_out, V)
    bcs_op.append(bc_gamma_out)

    dirichlet_bc = []
    if args.k in (0, 1, 2):
        # left Dirichlet boundary is active
        dirichlet_bc.append({
            "value": zero,
            "boundary": osp_config.gamma_d,
            "entity_dim": 1,
            "sub": 0})
    elif args.k in (8, 9, 10):
        # right Dirichlet boundary is active
        dirichlet_bc.append({
            "value": zero,
            "boundary": osp_config.gamma_d,
            "entity_dim": 0,
            "sub": 1})

    for bc_spec in dirichlet_bc:
        # determine entities and define BCTopo
        entities_omega = df.mesh.locate_entities_boundary(
            V.mesh, bc_spec["entity_dim"], bc_spec["boundary"]
        )
        entities_omega_in = df.mesh.locate_entities_boundary(
            V_in.mesh, bc_spec["entity_dim"], bc_spec["boundary"]
        )
        bc = BCTopo(
            df.fem.Constant(V.mesh, bc_spec["value"]),
            entities_omega,
            bc_spec["entity_dim"],
            V,
            sub=bc_spec["sub"],
        )
        bc_rp = BCTopo(
            df.fem.Constant(V_in.mesh, bc_spec["value"]),
            entities_omega_in,
            bc_spec["entity_dim"],
            V_in,
            sub=bc_spec["sub"],
        )
        bcs_op.append(bc)
        bcs_range_product.append(bc_rp)
    bcs_op = tuple(bcs_op)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))
    assert len(bcs_op) - 1 == len(bcs_range_product)

    # ### Auxiliary Problem
    # locate interfaces for definition of auxiliary problem
    left_most_cell = np.amin(cells_omega)
    unit_length = 1.0
    x_min = float(left_most_cell * unit_length)

    interface_locators = []
    for i in range(1, cells_omega.size):
        x_coord = float(x_min + i)
        interface_locators.append(plane_at(x_coord, "x"))
    aux_tags = list(range(15, 15 + cells_omega.size))

    if args.debug:
        for marker in interface_locators:
            entities = df.mesh.locate_entities(V.mesh, V.mesh.topology.dim-1, marker)
            assert entities.size == example.num_intervals
    assert len(aux_tags) == cells_omega.size
    assert len(interface_locators) == cells_omega.size - 1

    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    problem = LinearElasticityProblem(omega, V, phases=mat)
    params = Parameters({"R": cells_omega.size})

    # the auxiliary problem is supposed to be defined on Omega
    # but struct_grid is the coarse grid for the global domain!!!
    auxiliary_problem = GlobalAuxiliaryProblem(
        problem, aux_tags, params, struct_grid_omega, interface_locators=interface_locators
    )
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Discretize left hand side - FenicsxMatrixBasedOperator
    matparam = {"gdim": example.gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
    parageom = ParaGeomLinEla(
        omega,
        V,
        d=d_trafo,  # type: ignore
        matparam=matparam,
    )

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary_problem.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    # operator for left hand side on full oversampling domain
    operator = FenicsxMatrixBasedOperator(
        parageom.form_lhs, params, param_setter=param_setter, bcs=bcs_op
    )

    # ### Discretize right hand side - DirichletLift
    entities_gamma_out = df.mesh.locate_entities_boundary(
        V.mesh, V.mesh.topology.dim - 1, osp_config.gamma_out
    )
    expected_num_facets_gamma_out = (example.num_intervals - 2, 2 * (example.num_intervals - 2))
    assert entities_gamma_out.size in expected_num_facets_gamma_out
    rhs = DirichletLift(operator.range, operator.compiled_form, entities_gamma_out)  # type: ignore


    def l2(V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return ufl.inner(u, v) * ufl.dx # type: ignore

    # ### Source product operator
    l2_cpp = df.fem.form(l2(V))
    pmat_source = dolfinx.fem.petsc.create_matrix(l2_cpp)
    pmat_source.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_source, l2_cpp, bcs=[])
    pmat_source.assemble()
    source_mat = csr_array(pmat_source.getValuesCSR()[::-1])  # type: ignore
    source_product = NumpyMatrixOperator(source_mat[rhs.dofs, :][:, rhs.dofs], name="l2")

    # ### Range Product
    range_mat = LinearElasticMaterial(**matparam)
    linela_target = LinearElasticityProblem(omega_in, V_in, phases=range_mat)
    a_cpp = df.fem.form(linela_target.form_lhs)
    range_mat = dolfinx.fem.petsc.create_matrix(a_cpp)
    range_mat.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(range_mat, a_cpp, bcs=bcs_range_product)
    range_mat.assemble()
    range_product = FenicsxMatrixOperator(range_mat, V_in, V_in, name="energy")

    # ### Rigid body modes
    kernel_set = osp_config.kernel
    ns_vecs = build_nullspace(V_in, gdim=example.gdim)
    assert len(ns_vecs) == 3
    rigid_body_modes = []

    kernel = None
    if len(kernel_set) > 0:
        for j in kernel_set:
            dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
            rigid_body_modes.append(ns_vecs[j])
        kernel = source_V_in.make_array(rigid_body_modes)  # type: ignore
        gram_schmidt(kernel, product=None, copy=False)
        assert np.allclose(kernel.gramian(), np.eye(len(kernel)))
    assert kernel is not None

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        source_V_in,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
        padding=1e-8,
    )

    if osp_config.gamma_n is not None:
        assert omega.facet_tags.find(top_tag).size == example.num_intervals * 1  # top
        dA = ufl.Measure("ds", domain=omega.grid, subdomain_data=omega.facet_tags)
        t_y = -example.traction_y
        traction = df.fem.Constant(
            omega.grid,
            (df.default_scalar_type(0.0), df.default_scalar_type(t_y)),
        )
        v = ufl.TestFunction(V)
        L = ufl.inner(v, traction) * dA(facet_tag_definitions["top"][0])
        Lcpp = df.fem.form(L)
        f_ext = dolfinx.fem.petsc.create_vector(Lcpp)  # type: ignore

        with f_ext.localForm() as b_loc:
            b_loc.set(0)
        dolfinx.fem.petsc.assemble_vector(f_ext, Lcpp)

        # Apply boundary conditions to the rhs
        bcs_neumann = _create_dirichlet_bcs(bcs_op)
        dolfinx.fem.petsc.apply_lifting(f_ext, [operator.compiled_form], bcs=[bcs_neumann])  # type: ignore
        f_ext.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        dolfinx.fem.petsc.set_bc(f_ext, bcs_neumann)

        assert np.isclose(np.sum(f_ext.array), -example.traction_y)
        F_ext = operator.range.make_array([f_ext])  # type: ignore
    else:
        F_ext = operator.range.zeros(1)

    # ######### up to here its discretization of transfer and f_ext


    if args.debug:
        loglevel = 10
    else:
        loglevel = 20

    logfilename = example.log_basis_construction(args.nreal, "hapod", args.k).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("hapod", level=loglevel)

    # ### Generate training seed for each configuration
    # training_seeds = {}
    # for cfg, rndseed in zip(
    #     example.configurations,
    #     np.random.SeedSequence(example.training_set_seed).generate_state(len(example.configurations)),
    # ):
    #     training_seeds[cfg] = rndseed
    myseeds = np.random.SeedSequence(example.training_set_seed).generate_state(11)

    parameter_space = ParameterSpace(params, example.mu_range)
    parameter_name = "R"
    ntrain = example.ntrain(args.k)
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=myseeds[args.k],
    )
    logger.info("Starting range approximation of transfer operators" f" for training set of size {len(training_set)}.")

    # ### Generate random seed for each specific mu in the training set
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # start range approximation with `transfer` and `F_ext`
    require_neumann_data = np.any(np.nonzero(F_ext.to_numpy())[1])
    if 0 in cells_omega:
        assert require_neumann_data
    else:
        assert not require_neumann_data

    assert len(training_set) == len(seed_seqs_rrf)
    snapshots = transfer.range.empty()
    neumann_snapshots = transfer.range.empty(reserve=len(training_set))
    spectral_basis_sizes = list()

    epsilon_star = example.epsilon_star["hapod"]
    Nin = transfer.rhs.dofs.size
    epsilon_alpha = np.sqrt(1 - example.omega**2.0) * epsilon_star
    epsilon_pod = np.sqrt(ntrain) * example.omega * epsilon_star

    for mu, seed_seq in zip(training_set, seed_seqs_rrf):
        with new_rng(seed_seq):
            transfer.assemble_operator(mu)
            basis = adaptive_rrf_normal(
                logger,
                transfer,
                error_tol=example.rrf_ttol / example.l_char,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=Nin,
                l2_err=epsilon_alpha,
            )
            logger.info(f"\nSpectral Basis length: {len(basis)}.")
            spectral_basis_sizes.append(len(basis))
            snapshots.append(basis)  # type: ignore

            if require_neumann_data:
                logger.info("\nSolving for additional Neumann mode ...")
                U_neumann = transfer.op.apply_inverse(F_ext)
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
                neumann_snapshots.append(U_orth)  # type: ignore

    logger.info(f"Average length of spectral basis: {np.average(spectral_basis_sizes)}.")
    if len(neumann_snapshots) > 0:
        logger.info("Appending Neumann snapshots to global snapshot set.")
        snapshots.append(neumann_snapshots)

    logger.info("Computing final POD")
    spectral_modes, spectral_svals = pod(snapshots, product=transfer.range_product, l2_err=epsilon_pod)

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


if __name__ == "__main__":
    import sys
    import argparse

    # input arguments
    # tag for oversampling problem k = 0, ..., 10

    # factory for OversamplingConfig data class

    parser = argparse.ArgumentParser(
        description="Oversampling for ParaGeom example using HAPOD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("nreal", type=int, help="The n-th realization of the problem.")
    parser.add_argument("k", type=int, help="The oversampling problem for target subdomain Ω_in^k.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
