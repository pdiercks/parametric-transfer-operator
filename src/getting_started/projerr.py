from pathlib import Path

from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem
from dolfinx.fem.petsc import set_bc
from dolfinx.io import gmshio
import numpy as np

from multi.domain import RectangularDomain
from multi.projection import compute_absolute_proj_errors
from multi.product import InnerProduct
from multi.bcs import BoundaryConditions
from multi.solver import build_nullspace

from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem

    set_defaults(
        {
            "pymor.core.logger.getLogger.filename": beam.log_projerr(
                args.distribution, args.configuration
            ),
        }
    )
    logger = getLogger(Path(__file__).stem, level="INFO")

    # ### Unit cell domain
    meshfile = beam.unit_cell_grid
    domain, _, _ = gmshio.read_from_msh(
        meshfile.as_posix(), MPI.COMM_SELF, gdim=beam.gdim
    )
    omega = RectangularDomain(domain)

    # ### Beam Problem definitions
    beamproblem = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    cell_index = beamproblem.config_to_cell(args.configuration)
    dirichlet = beamproblem.get_dirichlet(cell_index)
    kernel_set = beamproblem.get_kernel_set(cell_index)
    logger.info(f"{kernel_set=}")

    # ### Translate the unit cell domain
    unit_length = omega.xmax[0] - omega.xmin[0]
    deltax = cell_index * unit_length
    dx = np.array([[deltax, 0.0, 0.0]])
    omega.translate(dx)

    fe = element("P", domain.basix_cell(), beam.fe_deg, shape=(beam.gdim,))
    V = fem.functionspace(domain, fe)
    source = FenicsxVectorSpace(V)

    # ### Range product operator
    # get homogeneous Dirichlet bcs if present
    bchandler = BoundaryConditions(domain, V)
    bc_hom = []
    if dirichlet is not None:
        bchandler.add_dirichlet_bc(**dirichlet)
        bc_hom = bchandler.bcs

    inner_product = InnerProduct(V, beam.range_product, bcs=bc_hom)
    pmat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(pmat, V, V)

    # ### Rigid body modes
    gdim = beam.gdim
    ns_vecs = build_nullspace(V, gdim=gdim)
    rigid_body_modes = []
    for j in kernel_set:
        set_bc(ns_vecs[j], bc_hom)
        rigid_body_modes.append(ns_vecs[j])
    basis = source.make_array(rigid_body_modes)

    basis_vectors = np.load(beam.loc_pod_modes(args.distribution, args.configuration))
    B = source.from_numpy(basis_vectors)
    basis.append(B)

    gram_schmidt(basis, product=product, copy=False)

    # ### load test data
    data = np.load(beam.fom_test_set(args.configuration))
    U = source.from_numpy(data)

    errs = compute_absolute_proj_errors(U, basis, product=product, orthonormal=True)
    np.save(beam.proj_error(args.distribution, args.configuration), errs)


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
    args = argparser.parse_args(sys.argv[1:])
    main(args)
