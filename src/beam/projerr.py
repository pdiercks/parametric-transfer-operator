from pathlib import Path

from mpi4py import MPI
from basix.ufl import element
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io import gmshio
import numpy as np

from multi.domain import RectangularSubdomain
from multi.projection import compute_relative_proj_errors
from multi.product import InnerProduct
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.shapes import NumpyLine

from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem

    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_projerr(
                args.distribution, args.configuration, args.name
            ),
        }
    )
    # FIXME logger not used
    logger = getLogger(Path(__file__).stem, level="INFO")

    # ### Unit cell domain
    meshfile = beam.unit_cell_grid
    domain, _, _ = gmshio.read_from_msh(
        meshfile.as_posix(), MPI.COMM_SELF, gdim=beam.gdim
    )
    omega = RectangularSubdomain(99, domain)

    # ### Beam Problem definitions
    beamproblem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    cell_index = beamproblem.config_to_cell(args.configuration)

    # ### Translate the unit cell domain
    unit_length = omega.xmax[0] - omega.xmin[0]
    deltax = cell_index * unit_length
    dx = np.array([[deltax, 0.0, 0.0]])
    omega.translate(dx)

    omega.create_coarse_grid(1)
    omega.create_boundary_grids()

    fe = element("P", domain.basix_cell(), beam.fe_deg, shape=(beam.gdim,))
    V = fem.functionspace(domain, fe)

    phases = LinearElasticMaterial(2, 20e3, 0.3)  # material will not be important here
    problem = LinElaSubProblem(omega, V, phases=(phases,))
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    # ### Load test data
    # test data contains full displacement solution restricted to edges
    npz_testset = np.load(beam.fom_test_set(args.configuration))
    npz_basis = np.load(
        beam.fine_scale_edge_modes_npz(args.distribution, args.configuration, args.name)
    )

    errors = {}

    for edge in ["bottom", "left", "right", "top"]:
        edge_space = problem.edge_spaces["fine"][edge]

        # BCs for range product
        facet_dim = edge_space.mesh.topology.dim - 1
        vertices = mesh.locate_entities_boundary(
            edge_space.mesh, facet_dim, lambda x: np.full(x[0].shape, True, dtype=bool)
        )
        _dofs = fem.locate_dofs_topological(edge_space, facet_dim, vertices)
        gdim = edge_space.mesh.geometry.dim
        bc_hom = fem.dirichletbc(
            np.array((0,) * gdim, dtype=default_scalar_type), _dofs, edge_space
        )

        # range product
        range_product = InnerProduct(edge_space, beam.range_product, bcs=[bc_hom])
        product_matrix = range_product.assemble_matrix()
        product = FenicsxMatrixOperator(product_matrix, edge_space, edge_space)

        # fine scale edge basis
        source = FenicsxVectorSpace(edge_space)
        basis = source.from_numpy(npz_basis[edge])
        G = basis.gramian(product)
        orthonormal = np.isclose(np.sum(G), len(basis))
        assert orthonormal

        # test data
        U = source.from_numpy(npz_testset[edge])

        # subtract coarse scale part
        if edge in ("bottom", "top"):
            component = 0
        else:
            component = 1
        # the order of `nodes` is important
        # cannot expect `dofs` to have correct ordering
        # that first points to `xmin` and then `xmax`
        # xmin = np.amin(edge_space.mesh.geometry.x, axis=0)
        # xmax = np.amax(edge_space.mesh.geometry.x, axis=0)
        # nodes = np.array([xmin[component], xmax[component]])
        xdofs = edge_space.tabulate_dof_coordinates()
        nodes = xdofs[_dofs, component]
        line = NumpyLine(nodes)
        shapes = line.interpolate(edge_space, component)
        coarse_basis = source.from_numpy(shapes)
        assert len(coarse_basis) == 2 * gdim
        dofs = bc_hom._cpp_object.dof_indices()[0]
        u_dofs = U.dofs(dofs)
        U -= coarse_basis.lincomb(u_dofs)
        assert np.isclose(np.sum(U.dofs(dofs)), 1e-9)

        # compute projection error for fine scale part
        errs = compute_relative_proj_errors(
            U, basis, product=product, orthonormal=orthonormal
        )
        errors[edge] = errs

    # TODO maybe have both absolute and relative errors written to disk?
    np.savez(beam.proj_error(args.distribution, args.configuration, args.name), **errors)


if __name__ == "__main__":
    import sys
    import argparse

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute projection error for given data set and reduced basis.",
    )
    argparser.add_argument(
        "distribution",
        type=str,
        help="Distribution that was used for sampling in the basis construction.",
    )
    argparser.add_argument(
        "configuration",
        type=str,
        help="Configuration of oversampling problem for which the test data should be read.",
        choices=("left", "inner", "right"),
    )
    argparser.add_argument(
        "name",
        type=str,
        help="Name of the training strategy used to compute the local basis.",
        choices=("hapod", "heuristic"),
    )
    args = argparser.parse_args(sys.argv[1:])
    main(args)
