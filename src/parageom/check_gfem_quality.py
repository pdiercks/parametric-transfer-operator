"""compute projection error to assess quality of the basis"""

from mpi4py import MPI
import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import ufl

import scipy.linalg as spla
from multi.domain import RectangularDomain
from multi.io import read_mesh
from multi.projection import project_array, relative_error, absolute_error

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer


def main(args):
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem
    from .fom import discretize_fom
    from .matrix_based_operator import _create_dirichlet_bcs, BCTopo
    from .dofmap_gfem import GFEMDofMap, select_modes, parageom_dof_distribution_factory

    assert args.archetype in (0, 1, 2, 3, 4)


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

    # translate unit cell for FOM solution restriction
    coarse_grid = global_auxp.coarse_grid
    cell_index = example.archetype_to_cell(args.archetype)
    cell_vertices = coarse_grid.get_entities(0, cell_index)
    x_vertices = coarse_grid.get_entity_coordinates(0, cell_vertices)
    dx = x_vertices[0]
    unit_cell = RectangularDomain(read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim":2})[0])
    unit_cell.translate(dx)
    V = df.fem.functionspace(unit_cell.grid, fom.solution_space.V.ufl_element())

    dofmap = GFEMDofMap(coarse_grid)

    # ### wrap as pymor objects
    source = FenicsxVectorSpace(V)
    max_dofs_per_vert_ = np.load(example.local_basis_dofs_per_vert(0, args.method, args.distr))
    max_dofs_per_vert = np.repeat(max_dofs_per_vert_, [1, 1, 6, 1, 1], axis=0)
    Nmax = {}
    Nmax["left"] = np.amax(max_dofs_per_vert[0])
    Nmax["inner"] = np.amax(max_dofs_per_vert[4])
    Nmax["right"] = np.amax(max_dofs_per_vert[9])
    dofs_per_vert = parageom_dof_distribution_factory(32, Nmax)
    dofmap.distribute_dofs(dofs_per_vert)

    basis_path = example.local_basis_npy(0, args.method, args.distr, args.archetype)
    local_basis = np.load(basis_path)
    rb = select_modes(local_basis, dofs_per_vert[cell_index], max_dofs_per_vert[cell_index])
    basis = source.from_numpy(rb)

    # ### Discretize range product (depends on configuration)
    config = example.cell_to_config(cell_index)
    hom_dirichlet = example.get_dirichlet(coarse_grid.grid, config)
    bcs_range_product = []
    if hom_dirichlet is not None:
        # determine entities and define BCTopo
        entities = df.mesh.locate_entities_boundary(
            V.mesh, hom_dirichlet["entity_dim"], hom_dirichlet["boundary"]
        )
        bc_rp = BCTopo(
            df.fem.Constant(V.mesh, hom_dirichlet["value"]),
            entities,
            hom_dirichlet["entity_dim"],
            V,
            sub=hom_dirichlet["sub"],
        )
        bcs_range_product.append(bc_rp)
    bcs_range_product = _create_dirichlet_bcs(tuple(bcs_range_product))

    def scaled_h1_0_semi(V, gdim, a=example.l_char):
        l_char = df.fem.Constant(V.mesh, df.default_scalar_type(a ** gdim))
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return l_char * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx # type: ignore

    h1_cpp = df.fem.form(scaled_h1_0_semi(V, example.gdim))
    pmat_range = dolfinx.fem.petsc.create_matrix(h1_cpp)
    pmat_range.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_range, h1_cpp, bcs=bcs_range_product)
    pmat_range.assemble()
    range_product = FenicsxMatrixOperator(pmat_range, V, V, name="scaled_h1_0_semi")

    # ### Add kernel (depends on configuration)
    # kernel_set = example.get_kernel_set(cell_index)
    # ### Rigid body modes
    # ns_vecs = build_nullspace(V, gdim=example.gdim)
    # rigid_body_modes = []
    # for j in kernel_set:
    #     dolfinx.fem.petsc.set_bc(ns_vecs[j], bcs_range_product)
    #     rigid_body_modes.append(ns_vecs[j])
    # full_basis = source.make_array(rigid_body_modes)  # type: ignore
    # gram_schmidt(full_basis, product=range_product, atol=0, rtol=0, copy=False)
    # print(full_basis.gramian(range_product))
    # full_basis.append(basis)

    # orthonormal = np.allclose(full_basis.gramian(range_product), np.eye(len(full_basis)), atol=1e-5)
    # if not orthonormal:
    #     raise ValueError("Basis is not orthonormal wrt range product.")

    # compute projection error
    pspace = fom.parameters.space(example.mu_range)
    test_set = pspace.sample_randomly(10)

    # mu_test = fom.parameters.parse([0.25479121, 0.18777569, 0.27171958, 0.23947361, 0.11883547,
    #    0.29512447, 0.25222794, 0.25721286, 0.12562273, 0.19007719])
    # test_set = [mu_test]

    u = df.fem.Function(fom.solution_space.V)
    uvec = u.x.petsc_vec
    u_local = df.fem.Function(V)
    u_local_vec = u_local.x.petsc_vec

    interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
            V.mesh, V.element, fom.solution_space.V.mesh, padding=1e-6)

    test_data = source.empty(reserve=len(test_set))
    for mu in test_set:
        # global FOM solution
        U_fom = fom.solve(mu)
        uvec.zeroEntries()
        uvec.array[:] = U_fom.to_numpy().flatten()
        u.x.scatter_forward()

        # restriction to target domain
        u_local_vec.zeroEntries()
        u_local.interpolate(u, nmm_interpolation_data=interp_data)
        u_local.x.scatter_forward()

        # wrap as pymor vectorarray
        U = source.from_numpy(u_local.x.array[:].reshape(1, -1))
        test_data.append(U)

    rerrs = []
    aerrs = []
    for N in range(len(basis) + 1):
    # N = len(basis)
        gramian = basis[:N].gramian(range_product)
        rhs = basis[:N].inner(test_data, range_product)
        coeffs = spla.solve(gramian, rhs, assume_a='pos', overwrite_a=True, overwrite_b=True).T
        U_proj = basis[:N].lincomb(coeffs)
        # U_proj = project_array(test_data, basis[:N], product=range_product, orthonormal=False)
        relerr = relative_error(test_data, U_proj, product=range_product)
        abserr = absolute_error(test_data, U_proj, product=range_product)
        rerrs.append(np.max(relerr))
        aerrs.append(np.max(abserr))

    # viz = FenicsxVisualizer(basis.space)
    # viz.visualize(U_proj, filename=f"recon_{cell_index}.xdmf")

    import matplotlib.pyplot as plt
    plt.semilogy(np.arange(len(rerrs)), rerrs, "r-")
    plt.show()
    breakpoint()

    if args.output is not None:
        np.save(args.output, np.array(aerrs))


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument("distr", type=str, help="Distribution used for random sampling.")
    parser.add_argument("archetype", type=int, help="Archetype.")
    parser.add_argument("--output", type=str, help="Write absolute projection error to file.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
