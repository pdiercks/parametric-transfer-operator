from mpi4py import MPI
from basix.ufl import element
from dolfinx import fem
from dolfinx.io import gmshio
import numpy as np

from multi.projection import compute_relative_proj_errors
from multi.product import InnerProduct

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator


def main(args):
    from .tasks import beam
    meshfile = beam.unit_cell_grid
    domain, _, _ = gmshio.read_from_msh(meshfile.as_posix(), MPI.COMM_SELF, gdim=beam.gdim)
    fe = element("P", domain.basix_cell(), beam.fe_deg, shape=(beam.gdim, ))
    V = fem.functionspace(domain, fe)
    source = FenicsxVectorSpace(V)

    data = np.load(beam.fom_test_set(args.configuration))
    U = source.from_numpy(data)

    basis_vectors = np.load(beam.loc_pod_modes(args.distribution, args.configuration))
    basis = source.from_numpy(basis_vectors)

    inner_product = InnerProduct(V, beam.range_product, bcs=[])
    product_mat = inner_product.assemble_matrix()
    product = FenicsxMatrixOperator(product_mat, V, V)

    gram_schmidt(basis, product=product, copy=False)

    errs = compute_relative_proj_errors(U, basis, product=product, orthonormal=True)
    np.save(beam.proj_error(args.distribution, args.configuration), errs)



if __name__ == "__main__":
    import sys
    import argparse

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute projection error for given data set and reduced basis.",
    )
    argparser.add_argument("distribution", type=str, help="Distribution that was used for sampling in the basis construction.")
    argparser.add_argument("configuration", type=str, help="Configuration of oversampling problem for which the test data should be read.", choices=("left", "inner", "right"))
    args = argparser.parse_args(sys.argv[1:])
    main(args)
