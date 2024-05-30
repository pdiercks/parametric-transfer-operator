"""compute projection error to assess quality of the basis"""

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np

from multi.io import read_mesh
from multi.interpolation import make_mapping
from multi.projection import project_array, relative_error, absolute_error
from multi.product import InnerProduct
from multi.solver import build_nullspace

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def main(args):
    from .tasks import example
    from .definitions import BeamProblem
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom
    from .matrix_based_operator import _create_dirichlet_bcs, BCGeom, BCTopo

    cell_index = example.config_to_cell(args.config)

    # ### Discretize global FOM
    global_parent_domain_path = example.parent_domain("global").as_posix()
    coarse_grid_path = example.coarse_grid("global").as_posix()
    # do not change interface tags; see src/parageom/preprocessing.py::create_parent_domain
    interface_tags = [i for i in range(15, 25)]
    global_auxp = discretize_auxiliary_problem(
        global_parent_domain_path,
        example.geom_deg,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    trafo_d_gl = df.fem.Function(global_auxp.problem.V, name="d_trafo")
    fom = discretize_fom(example, global_auxp, trafo_d_gl)

    # extract u for single cell
    beam_problem = BeamProblem(
        example.coarse_grid("global"), example.parent_domain("global"), example
    )
    delta_x = beam_problem.get_xmin_omega_in(cell_index)

    unit_cell, _, _ = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim":2})
    x_geom = unit_cell.geometry.x
    x_geom += delta_x
    V = df.fem.functionspace(unit_cell, fom.solution_space.V.ufl_element())

    dofs = make_mapping(V, fom.solution_space.V)

    # ### wrap as pymor objects
    source = FenicsxVectorSpace(V)
    basis_path = None
    if args.method == "hapod":
        basis_path = example.hapod_modes_npy(0, args.distr, args.config)
    elif args.method == "heuristic":
        basis_path = example.heuristic_modes_npy(0, args.distr, args.config)
    else:
        raise NotImplementedError
    local_basis = np.load(basis_path)
    basis = source.from_numpy(local_basis)

    # ### Discretize range product (depends on configuration)
    hom_dirichlet = beam_problem.get_dirichlet(cell_index)
    bcs_range_product = []
    if hom_dirichlet is not None:
        sub = hom_dirichlet.get("sub", None)
        if sub is not None:
            # determine entities and define BCTopo
            entities = df.mesh.locate_entities_boundary(
                V.mesh, hom_dirichlet["entity_dim"], hom_dirichlet["boundary"]
            )
            bc_rp = BCTopo(
                hom_dirichlet["value"],
                entities,
                hom_dirichlet["entity_dim"],
                V,
                sub=sub,
            )
        else:
            bc_rp = BCGeom(hom_dirichlet["value"], hom_dirichlet["boundary"], V)
        bcs_range_product.append(bc_rp)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))

    inner_product = InnerProduct(V, example.range_product, bcs=bcs_range_product)
    pmat = inner_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(pmat, V, V)

    # ### Add kernel (depends on configuration)
    kernel_set = beam_problem.get_kernel_set(cell_index)
    # ### Rigid body modes
    ns_vecs = build_nullspace(V, gdim=example.gdim)
    rigid_body_modes = []
    for j in kernel_set:
        dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
        rigid_body_modes.append(ns_vecs[j])
    full_basis = source.make_array(rigid_body_modes)  # type: ignore
    gram_schmidt(full_basis, product=range_product, atol=0, rtol=0, copy=False)
    full_basis.append(basis)

    orthonormal = np.allclose(full_basis.gramian(range_product), np.eye(len(full_basis)))
    if orthonormal:
        print("basis is orthonormal")
    print(f"basis length: {len(full_basis)}")

    # compute projection error
    pspace = fom.parameters.space(example.mu_range)
    test_set = pspace.sample_randomly(10)

    test_data = source.empty(reserve=len(test_set))
    for mu in test_set:
        U_fom = fom.solve(mu)
        u_local = U_fom.dofs(dofs)
        U = source.from_numpy(u_local)
        test_data.append(U)

    rerrs = []
    aerrs = []
    for N in range(len(full_basis) + 1):
        U_proj = project_array(test_data, full_basis[:N], product=range_product, orthonormal=orthonormal)
        relerr = relative_error(test_data, U_proj, product=range_product)
        abserr = absolute_error(test_data, U_proj, product=range_product)
        rerrs.append(np.max(relerr))
        aerrs.append(np.max(abserr))

    if args.output is not None:
        np.save(args.output, np.array(aerrs))


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument("distr", type=str, help="Distribution used for random sampling.")
    parser.add_argument("config", type=str, help="Configuration / Archetype.")
    parser.add_argument("--output", type=str, help="Write absolute projection error to file.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
