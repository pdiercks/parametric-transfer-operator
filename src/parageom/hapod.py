# from itertools import repeat
from pathlib import Path

# import concurrent.futures
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from petsc4py import PETSc
from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import ufl

from pymor.bindings.fenicsx import (
    FenicsxVectorSpace,
    FenicsxMatrixOperator,
    FenicsxVisualizer,
)
from pymor.algorithms.pod import pod
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.random import new_rng
from pymor.parameters.base import ParameterSpace

from multi.io import read_mesh
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.solver import build_nullspace


def adaptive_rrf_normal(
    logger,
    transfer_problem,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    **sampling_options,
):
    r"""Adaptive randomized range approximation of `A`.
    This is an implementation of Algorithm 1 in [BS18]_.

    Given the |Operator| `A`, the return value of this method is the
    |VectorArray| `B` with the property

    .. math::
        \Vert A - P_{span(B)} A \Vert \leq tol

    with a failure probability smaller than `failure_tolerance`, where the
    norm denotes the operator norm. The inner product of the range of
    `A` is given by `range_product` and
    the inner product of the source of `A` is given by `source_product`.

    NOTE
    ----
    Instead of a transfer operator A, a transfer problem is used.
    (see multi.problem.TransferProblem)
    The image Av = A.apply(v) is equivalent to the restriction
    of the full solution to the target domain Ω_in, i.e.
        U = transfer_problem.solve(v)

    Parameters
    ----------
    transfer_problem
        The transfer problem associated with a (transfer) |Operator| A.
    source_product
        Inner product |Operator| of the source of A.
    range_product
        Inner product |Operator| of the range of A.
    error_tol
        Error tolerance for the algorithm.
    failure_tolerance
        Maximum failure probability.
    num_testvecs
        Number of test vectors.
    lambda_min
        The smallest eigenvalue of source_product.
        If `None`, the smallest eigenvalue is computed using scipy.
    sampling_options
        Optional keyword arguments for the generation of
        random samples (training data).
        see `_create_random_values`.

    Returns
    -------
    B
        |VectorArray| which contains the basis, whose span approximates the range of A.

    """

    tp = transfer_problem
    distribution = "normal"

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
        lambda_min = eigsh(
            L, sigma=0, which="LM", return_eigenvectors=False, k=1, OPinv=Linv
        )[0]

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = tp.rhs.dofs.size
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )

    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution)
    M = tp.solve(R)
    B = tp.range.empty()
    maxnorm = np.inf
    while maxnorm > testlimit:
        basis_length = len(B)
        v = tp.generate_random_boundary_data(1, distribution, **sampling_options)

        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        logger.debug(f"{maxnorm=}")

    return B


def main(args):
    from .tasks import example
    from .definitions import BeamProblem
    from .lhs import sample_lhs
    from .auxiliary_problem import GlobalAuxiliaryProblem
    from .locmor import ParametricTransferProblem, DirichletLift
    from .matrix_based_operator import (
        FenicsxMatrixBasedOperator,
        BCGeom,
        BCTopo,
        _create_dirichlet_bcs,
    )
    from .fom import ParaGeomLinEla

    method = Path(__file__).stem  # hapod
    logfilename = example.log_basis_construction(
        args.nreal, method, args.distribution, args.configuration
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level="DEBUG")

    # ### Generate training seed for each configuration
    training_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.training_set_seed).generate_state(
            len(example.configurations)
        ),
    ):
        training_seeds[cfg] = rndseed

    parameter_space = ParameterSpace(
        example.parameters[args.configuration], example.mu_range
    )
    parameter_name = list(example.parameters[args.configuration].keys())[0]
    ntrain = example.ntrain(args.configuration)
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=training_seeds[args.configuration],
    )
    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
    )

    # with concurrent.futures.ProcessPoolExecutor(
    #     max_workers=args.max_workers
    # ) as executor:
    #     results = executor.map(
    #         spawn_rng(approximate_range),
    #         repeat(beam),
    #         training_set,
    #         repeat(args.configuration),
    #         repeat(args.distribution),
    #     )

    # ### Generate random seed for each specific mu in the training set
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(ntrain)

    # ### Oversampling Domain
    domain, ct, ft = read_mesh(
        example.parent_domain(args.configuration), MPI.COMM_SELF, gdim=example.gdim
    )
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)
    ft_def = {"bottom": int(11), "left": int(12), "right": int(13), "top": int(14)}
    omega.create_facet_tags(ft_def)

    aux_tags = None
    if args.configuration == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]

    elif args.configuration == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        aux_tags = [15, 16]

    elif args.configuration == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        aux_tags = [15, 16]
    else:
        raise NotImplementedError

    # ### Structured coarse grid
    grid, _, _ = read_mesh(
        example.coarse_grid(args.configuration), MPI.COMM_SELF, gdim=example.gdim
    )
    coarse_grid = StructuredQuadGrid(grid)

    # ### Auxiliary Problem
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu)
    V = df.fem.functionspace(omega.grid, ("P", example.geom_deg, (example.gdim,)))
    problem = LinearElasticityProblem(omega, V, phases=mat)
    auxiliary_problem = GlobalAuxiliaryProblem(
        problem, aux_tags, example.parameters[args.configuration], coarse_grid
    )
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Beam Problem
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.parent_domain("global"), example
    )
    cell_index = beam_problem.config_to_cell(args.configuration)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    hom_dirichlet = beam_problem.get_dirichlet(cell_index)
    kernel_set = beam_problem.get_kernel_set(cell_index)

    # ### Target subdomain & Range space
    xmin_omega_in = beam_problem.get_xmin_omega_in(cell_index)
    logger.debug(f"{xmin_omega_in=}")
    target_domain, _, _ = read_mesh(
        example.parent_unit_cell, MPI.COMM_SELF, gdim=example.gdim
    )
    omega_in = RectangularDomain(target_domain)
    omega_in.translate(xmin_omega_in)
    logger.debug(f"{omega_in.xmin=}")
    V_in = df.fem.functionspace(target_domain, V.ufl_element())
    target_space = FenicsxVectorSpace(V_in)

    # create necessary connectivities
    omega.grid.topology.create_connectivity(0, 2)
    omega_in.grid.topology.create_connectivity(0, 2)

    # ### Dirichlet BCs
    # have to be defined twice (operator & range product)
    zero = df.fem.Constant(V.mesh, (df.default_scalar_type(0.0),) * example.gdim)
    bc_gamma_out = BCGeom(zero, gamma_out, V)
    bcs_op = list()
    bcs_op.append(bc_gamma_out)
    bcs_range_product = []
    if hom_dirichlet is not None:
        sub = hom_dirichlet.get("sub", None)
        if sub is not None:
            # determine entities and define BCTopo
            entities = df.mesh.locate_entities_boundary(
                V.mesh, hom_dirichlet["entity_dim"], hom_dirichlet["boundary"]
            )
            bc = BCTopo(
                hom_dirichlet["value"],
                entities,
                hom_dirichlet["entity_dim"],
                V,
                sub=sub,
            )
            bc_rp = BCTopo(
                hom_dirichlet["value"],
                entities,
                hom_dirichlet["entity_dim"],
                V_in,
                sub=sub,
            )
        else:
            bc = BCGeom(hom_dirichlet["value"], hom_dirichlet["boundary"], V)
            bc_rp = BCGeom(hom_dirichlet["value"], hom_dirichlet["boundary"], V_in)
        bcs_op.append(bc)
        bcs_range_product.append(bc_rp)
    bcs_op = tuple(bcs_op)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))
    assert len(bcs_op) - 1 == len(bcs_range_product)

    # ### FenicsxMatrixBasedOperator
    parageom = ParaGeomLinEla(
            omega, V, E=example.youngs_modulus, NU=example.poisson_ratio, d=d_trafo # type: ignore
    )  # type: ignore
    params = example.parameters[args.configuration]

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary_problem.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    operator = FenicsxMatrixBasedOperator(
        parageom.form_lhs, params, param_setter=param_setter, bcs=bcs_op
    )

    # ### DirichletLift
    entities_gamma_out = df.mesh.locate_entities_boundary(
        V.mesh, V.mesh.topology.dim - 1, gamma_out
    )
    rhs = DirichletLift(operator.range, operator.compiled_form, entities_gamma_out)  # type: ignore

    # ### Range product operator
    inner_product = InnerProduct(V_in, example.range_product, bcs=bcs_range_product)
    pmat = inner_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(pmat, V_in, V_in)

    # ### Source product operator
    inner_source = InnerProduct(V, "l2", bcs=[])
    source_mat = csr_array(inner_source.assemble_matrix().getValuesCSR()[::-1])  # type: ignore
    source_product = NumpyMatrixOperator(source_mat[rhs.dofs, :][:, rhs.dofs])

    # ### Rigid body modes
    ns_vecs = build_nullspace(V_in, gdim=omega_in.grid.geometry.dim)
    rigid_body_modes = []
    for j in kernel_set:
        dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
        rigid_body_modes.append(ns_vecs[j])
    kernel = target_space.make_array(rigid_body_modes)  # type: ignore
    with logger.block("Orthonormalizing kernel of A ..."):  # type: ignore
        gram_schmidt(kernel, product=range_product, copy=False)

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        target_space,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
    )

    # ### Discretize Neumann Data
    dA = ufl.Measure('ds', domain=omega.grid, subdomain_data=omega.facet_tags)
    traction = df.fem.Constant(omega.grid, (df.default_scalar_type(0.0), df.default_scalar_type(-example.traction_y)))
    v = ufl.TestFunction(V)
    L = ufl.inner(v, traction) * dA(ft_def["top"])
    Lcpp = df.fem.form(L)
    f_ext = dolfinx.fem.petsc.create_vector(Lcpp) # type: ignore

    with f_ext.localForm() as b_loc:
        b_loc.set(0)
    dolfinx.fem.petsc.assemble_vector(f_ext, Lcpp)

    # Apply boundary conditions to the rhs
    bcs_neumann = _create_dirichlet_bcs(bcs_op)
    dolfinx.fem.petsc.apply_lifting(f_ext, [operator.compiled_form], bcs=[bcs_neumann])  # type: ignore
    f_ext.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    dolfinx.fem.petsc.set_bc(f_ext, bcs_neumann)
    FEXT = operator.range.make_array([f_ext]) # type: ignore

    assert len(training_set) == len(seed_seqs_rrf)
    snapshots = transfer.range.empty()

    for mu, seed_seq in zip(training_set, seed_seqs_rrf):
        with new_rng(seed_seq):
            # FIXME: transfer.assemble_operator(mu) is called where? when?
            transfer.assemble_operator(mu)
            basis = adaptive_rrf_normal(
                logger,
                transfer,
                error_tol=example.rrf_ttol,
                failure_tolerance=example.rrf_ftol,
                num_testvecs=example.rrf_num_testvecs,
            )
            logger.info(f"\nSpectral Basis length: {len(basis)}.")
            logger.info("\nSolving for additional Neumann mode ...")

            U_neumann = transfer.op.apply_inverse(FEXT)
            u_vec = transfer._u.x.petsc_vec # type: ignore
            u_vec.array[:] = U_neumann.to_numpy().flatten()
            transfer._u.x.scatter_forward() # type: ignore

            # ### restrict full solution to target subdomain
            transfer._u_in.interpolate(transfer._u, nmm_interpolation_data=transfer._interp_data) # type: ignore
            transfer._u_in.x.scatter_forward() # type: ignore
            U_in_neumann = transfer.range.make_array([transfer._u_in.x.petsc_vec.copy()]) # type: ignore

            # ### Remove kernel after restriction to target subdomain
            U_orth = orthogonal_part(U_in_neumann, kernel, product=transfer.range_product, orthonormal=True)
            basis.append(U_orth)

        snapshots.append(basis) # type: ignore

    pod_modes, pod_svals = pod(snapshots, product=transfer.range_product, l2_err=example.pod_l2_err)  # type: ignore

    viz = FenicsxVisualizer(pod_modes.space)
    viz.visualize(
        pod_modes,
        filename=example.hapod_modes_xdmf(
            args.nreal, args.distribution, args.configuration
        ),
    )
    np.save(
        example.hapod_modes_npy(args.nreal, args.distribution, args.configuration),
        pod_modes.to_numpy(),
    )
    np.save(
        example.hapod_singular_values(
            args.nreal, args.distribution, args.configuration
        ),
        pod_svals,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Computes fine scale edge basis functions via transfer problems and subsequently the POD of these sets of basis functions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "distribution", type=str, help="The distribution to draw samples from."
    )
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    parser.add_argument("nreal", type=int, help="The n-th realization of the problem.")
    parser.add_argument(
        "--max_workers", type=int, default=4, help="The max number of workers."
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
