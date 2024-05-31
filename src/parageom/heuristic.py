from pathlib import Path

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
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.core.defaults import set_defaults
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.random import new_rng, get_rng
from pymor.parameters.base import ParameterSpace

from multi.io import read_mesh
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct
from multi.projection import orthogonal_part
from multi.solver import build_nullspace


def heuristic_range_finder(
    logger,
    transfer_problem,
    training_set,
    testing_set,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    **sampling_options,
):
    """Heuristic range approximation."""

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

    # ### Compute non-parametric testlimit
    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = tp.rhs.dofs.size
    testfail = failure_tolerance / min(num_source_dofs, tp.range.dim)
    testlimit = (
        np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol
    )
    logger.info(f"{lambda_min=}")
    logger.info(f"{testlimit=}")

    logger.debug(f"Computing test set of size {len(testing_set) * num_testvecs}.")
    M = tp.range.empty()  # global test set
    for mu in testing_set:
        tp.assemble_operator(mu)
        R = tp.generate_random_boundary_data(
            count=num_testvecs, distribution=distribution
        )
        M.append(tp.solve(R))

    rng = get_rng()  # current RNG
    training_samples = []  # parameter values used in the training
    B = tp.range.empty()
    maxnorm = np.inf
    num_iter = 0
    while maxnorm > testlimit:
        basis_length = len(B)
        # randomly select mu from existing LHS design
        mu_ind = rng.integers(0, len(training_set))
        logger.debug(f"{mu_ind=}")
        mu = training_set.pop(mu_ind)
        training_samples.append(mu)
        tp.assemble_operator(mu)
        v = tp.generate_random_boundary_data(1, distribution, **sampling_options)

        B.append(tp.solve(v))
        gram_schmidt(B, range_product, atol=0, rtol=0, offset=basis_length, copy=False)
        M -= B.lincomb(B.inner(M, range_product).T)
        maxnorm = np.max(M.norm(range_product))
        num_iter += 1
        logger.debug(f"{num_iter=}\t{maxnorm=}")
    logger.info(f"Finished heuristic range approx. in {num_iter} iterations.")

    return B, training_samples


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

    method = Path(__file__).stem  # heuristic
    logfilename = example.log_basis_construction(
        args.nreal, method, args.distribution, args.configuration
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(method, level="DEBUG")

    # ### Generate training and testing seed for each configuration
    training_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.training_set_seed).generate_state(
            len(example.configurations)
        ),
    ):
        training_seeds[cfg] = rndseed
    testing_seeds = {}
    for cfg, rndseed in zip(
        example.configurations,
        np.random.SeedSequence(example.testing_set_seed).generate_state(
            len(example.configurations)
        ),
    ):
        testing_seeds[cfg] = rndseed

    parameter_space = ParameterSpace(
        example.parameters[args.configuration], example.mu_range
    )
    parameter_name = list(example.parameters[args.configuration].keys())[0]
    ntrain = example.ntrain(args.configuration)

    # FIXME
    # I am not sure if I should simply sample mu randomly inside the while loop
    # `samples` will have an influence on the LHS design
    # but still this should be better than walk through parameter space randomly?
    # the downside is that I cannot guarantee that the training set will be larger
    # than the number of iterations required
    training_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=4 * ntrain,
        criterion="center",
        random_state=training_seeds[args.configuration],
    )

    # NOTE
    # The testing set for the heuristic range approximation should have
    # the same size as the training set in the HAPOD, such that the
    # range of the same number of different transfer operators is approximated
    # in both variants.
    testing_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=ntrain,
        criterion="center",
        random_state=testing_seeds[args.configuration],
    )
    logger.info(
        "Starting range approximation of transfer operators"
        f" for training set of size {len(training_set)}."
    )

    # ### Generate random seed to draw random samples in the range finder algorithm
    # in the case of the heuristic range finder there is only one while loop
    realizations = np.load(example.realizations)
    this = realizations[args.nreal]
    seed_seqs_rrf = np.random.SeedSequence(this).generate_state(1)

    # ### Oversampling Domain
    domain, ct, ft = read_mesh(
        example.parent_domain(args.configuration),
        MPI.COMM_SELF,
        kwargs={"gdim": example.gdim},
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
        example.coarse_grid(args.configuration),
        MPI.COMM_SELF,
        kwargs={"gdim": example.gdim},
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
        example.parent_unit_cell, MPI.COMM_SELF, kwargs={"gdim": example.gdim}
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
        # determine entities and define BCTopo
        entities_omega = df.mesh.locate_entities_boundary(
            V.mesh, hom_dirichlet["entity_dim"], hom_dirichlet["boundary"]
        )
        entities_omega_in = df.mesh.locate_entities_boundary(
            V_in.mesh, hom_dirichlet["entity_dim"], hom_dirichlet["boundary"]
        )
        bc = BCTopo(
            df.fem.Constant(V.mesh, hom_dirichlet["value"]),
            entities_omega,
            hom_dirichlet["entity_dim"],
            V,
            sub=hom_dirichlet["sub"],
        )
        bc_rp = BCTopo(
            df.fem.Constant(V_in.mesh, hom_dirichlet["value"]),
            entities_omega_in,
            hom_dirichlet["entity_dim"],
            V_in,
            sub=hom_dirichlet["sub"],
        )
        bcs_op.append(bc)
        bcs_range_product.append(bc_rp)
    bcs_op = tuple(bcs_op)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))
    assert len(bcs_op) - 1 == len(bcs_range_product)

    # ### FenicsxMatrixBasedOperator
    parageom = ParaGeomLinEla(
        omega,
        V,
        E=example.youngs_modulus,
        NU=example.poisson_ratio,
        d=d_trafo,  # type: ignore
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
    ns_vecs = build_nullspace(V_in, gdim=example.gdim)
    assert len(ns_vecs) == 3
    rigid_body_modes = []
    for j in kernel_set:
        dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
        rigid_body_modes.append(ns_vecs[j])
    kernel = target_space.make_array(rigid_body_modes)  # type: ignore
    with logger.block("Orthonormalizing kernel of A ..."):  # type: ignore
        gram_schmidt(kernel, product=range_product, copy=False)
    assert np.allclose(kernel.gramian(range_product), np.eye(len(kernel)))

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        target_space,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
        padding=1e-6,
    )

    # ### Discretize Neumann Data
    dA = ufl.Measure("ds", domain=omega.grid, subdomain_data=omega.facet_tags)
    traction = df.fem.Constant(
        omega.grid,
        (df.default_scalar_type(0.0), df.default_scalar_type(-example.traction_y)),
    )
    v = ufl.TestFunction(V)
    L = ufl.inner(v, traction) * dA(ft_def["top"])
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
    FEXT = operator.range.make_array([f_ext])  # type: ignore

    # ### Heuristic range approximation
    training_set_length = len(training_set)
    logger.debug(f"{seed_seqs_rrf[0]=}")
    with new_rng(seed_seqs_rrf[0]):
        spectral_basis, training_samples = heuristic_range_finder(
            logger,
            transfer,
            training_set,
            testing_set,
            error_tol=example.rrf_ttol,
            failure_tolerance=example.rrf_ftol,
            num_testvecs=example.rrf_num_testvecs,
        )
    assert len(training_set) + len(training_samples) == training_set_length

    # ### Compute Neumann Modes
    neumann_snapshots = spectral_basis.space.empty(reserve=len(training_samples))
    for mu in training_samples:
        transfer.assemble_operator(mu)
        U_neumann = transfer.op.apply_inverse(FEXT)
        u_vec = transfer._u.x.petsc_vec  # type: ignore
        u_vec.array[:] = U_neumann.to_numpy().flatten()
        transfer._u.x.scatter_forward()  # type: ignore

        # ### restrict full solution to target subdomain
        transfer._u_in.interpolate(
            transfer._u, nmm_interpolation_data=transfer._interp_data
        )  # type: ignore
        transfer._u_in.x.scatter_forward()  # type: ignore
        U_in_neumann = transfer.range.make_array([transfer._u_in.x.petsc_vec.copy()])  # type: ignore

        # ### Remove kernel after restriction to target subdomain
        U_orth = orthogonal_part(
            U_in_neumann, kernel, product=transfer.range_product, orthonormal=True
        )
        neumann_snapshots.append(U_orth)

    assert np.allclose(
        spectral_basis.gramian(transfer.range_product), np.eye(len(spectral_basis))
    )
    logger.info("Extending spectral basis by Neumann snapshots ...")
    # U_proj_err = neumann_snapshots - spectral_basis.lincomb(
    #     neumann_snapshots.inner(spectral_basis, transfer.range_product)
    # )
    neumann_modes = pod(
        neumann_snapshots,
        modes=len(neumann_snapshots),
        product=transfer.range_product,
        l2_err=example.pod_l2_err,
        orth_tol=np.inf,
    )[0]

    basis_length = len(spectral_basis)
    spectral_basis.append(neumann_modes)
    gram_schmidt(
        spectral_basis,
        offset=basis_length,
        product=transfer.range_product,
        copy=False,
        check=False,
    )
    assert np.allclose(spectral_basis.gramian(transfer.range_product),
                       np.eye(len(spectral_basis)))

    logger.info(f"Spectral basis size: {basis_length}.")
    logger.info(f"Neumann modes: {len(neumann_modes)}/{len(training_samples)}.")
    logger.info(f"Final basis length: {len(spectral_basis)}.")

    viz = FenicsxVisualizer(spectral_basis.space)
    viz.visualize(
        spectral_basis,
        filename=example.heuristic_modes_xdmf(
            args.nreal, args.distribution, args.configuration
        ),
    )
    np.save(
        example.heuristic_modes_npy(args.nreal, args.distribution, args.configuration),
        spectral_basis.to_numpy(),
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
    args = parser.parse_args(sys.argv[1:])
    main(args)
