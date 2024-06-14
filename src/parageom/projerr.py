"""compute projection error to assess quality of the basis"""

from pathlib import Path

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np

from multi.io import read_mesh
from multi.projection import project_array, relative_error
from multi.product import InnerProduct
from multi.solver import build_nullspace
from multi.domain import RectangularDomain, StructuredQuadGrid
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator
from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.tools.random import new_rng

from scipy.sparse import csr_array


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

    logfilename = example.log_projerr(
        args.nreal, args.method, args.distr, args.config
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("projerr", level="INFO")

    # ### Oversampling Domain
    domain, ct, ft = read_mesh(
        example.parent_domain(args.config),
        MPI.COMM_SELF,
        kwargs={"gdim": example.gdim},
    )
    omega = RectangularDomain(domain, cell_tags=ct, facet_tags=ft)
    ft_def = {"bottom": int(11), "left": int(12), "right": int(13), "top": int(14)}
    omega.create_facet_tags(ft_def)

    aux_tags = None
    if args.config == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 4  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 4  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        assert omega.facet_tags.find(18).size == example.num_intervals * 4  # void 4
        aux_tags = [15, 16, 17, 18]

    elif args.config == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]

    elif args.config == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3  # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1  # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1  # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3  # top
        assert omega.facet_tags.find(15).size == example.num_intervals * 4  # void 1
        assert omega.facet_tags.find(16).size == example.num_intervals * 4  # void 2
        assert omega.facet_tags.find(17).size == example.num_intervals * 4  # void 3
        aux_tags = [15, 16, 17]
    else:
        raise NotImplementedError

    # ### Structured coarse grid
    grid, _, _ = read_mesh(
        example.coarse_grid(args.config),
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
        problem, aux_tags, example.parameters[args.config], coarse_grid
    )
    d_trafo = df.fem.Function(V, name="d_trafo")

    # ### Beam Problem
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.parent_domain("global"), example
    )
    cell_index = beam_problem.config_to_omega_in(args.config)[0]
    # target subdomain needs to be translated by lower left corner point
    assert cell_index in (0, 4, 8)
    gamma_out = beam_problem.get_gamma_out(cell_index)
    hom_dirichlet = beam_problem.get_dirichlet(cell_index)
    kernel_set = beam_problem.get_kernel_set(cell_index)

    # ### Target subdomain & Range space
    xmin_omega_in = beam_problem.get_xmin(cell_index)
    # logger.debug(f"{xmin_omega_in=}")
    target_domain, _, _ = read_mesh(
        example.target_subdomain, MPI.COMM_SELF, kwargs={"gdim": example.gdim}
    )
    omega_in = RectangularDomain(target_domain)
    omega_in.translate(xmin_omega_in)
    # logger.debug(f"{omega_in.xmin=}")
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
        E=1.,
        NU=example.poisson_ratio,
        d=d_trafo,  # type: ignore
    )  # type: ignore
    params = example.parameters[args.config]

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
        padding=1e-8,
    )

    # ### Read basis and wrap as pymor object
    basis_path = None
    if args.method == "hapod":
        basis_path = example.hapod_modes_npy(args.nreal, args.distr, args.config)
    elif args.method == "heuristic":
        basis_path = example.heuristic_modes_npy(args.nreal, args.distr, args.config)
    else:
        raise NotImplementedError
    local_basis = np.load(basis_path)
    basis = transfer.range.from_numpy(local_basis)

    full_basis = transfer.range.make_array(rigid_body_modes)  # type: ignore
    gram_schmidt(full_basis, product=range_product, atol=0, rtol=0, copy=False)
    logger.debug(full_basis.gramian(range_product))
    full_basis.append(basis)

    orthonormal = np.allclose(full_basis.gramian(range_product), np.eye(len(full_basis)), atol=1e-5)
    if not orthonormal:
        raise ValueError("Basis is not orthonormal wrt range product.")

    # Definition of validation set
    # make sure that this is always the same set of parameters
    # and also same set of boundary data
    # but different from μ and g used in the training
    parameter_space = operator.parameters.space(example.mu_range)
    parameter_name = list(example.parameters[args.config].keys())[0]
    test_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=30,
        criterion="center",
        random_state=example.projerr_seed
    )

    test_data = transfer.range.empty(reserve=len(test_set))

    logger.info(f"Computing test set of size {len(test_set)}...")
    with new_rng(example.projerr_seed):
        for mu in test_set:
            transfer.assemble_operator(mu)
            g = transfer.generate_random_boundary_data(1, "normal", {"scale": 0.1})
            test_data.append(transfer.solve(g))

    aerrs = []
    rerrs = []
    u_norm = test_data.norm(range_product) # norm of each test vector

    logger.info("Computing relative projection error ...")
    for N in range(len(full_basis) + 1):
        U_proj = project_array(test_data, full_basis[:N], product=range_product, orthonormal=orthonormal)
        err = (test_data - U_proj).norm(range_product) # absolute projection error
        if np.all(err == 0.):
            # ensure to return 0 here even when the norm of U is zero
            rel_err = err
        else:
            rel_err = err / u_norm
        aerrs.append(np.max(err))
        rerrs.append(np.max(rel_err))

    rerr = np.array(rerrs)
    aerr = np.array(aerrs)
    if args.output is not None:
        np.save(args.output, np.vstack((rerr, aerr)).T)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=str, help="The n-th realization.")
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument("distr", type=str, help="Distribution used for random sampling.")
    parser.add_argument("config", type=str, help="Configuration / Archetype.")
    parser.add_argument("--output", type=str, help="Write absolute and relative projection error to file.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
