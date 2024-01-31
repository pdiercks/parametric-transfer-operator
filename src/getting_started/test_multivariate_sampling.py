
import numpy as np
from scipy.linalg import eigh
# from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import eigsh

from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element
from multi.bcs import get_boundary_dofs
from multi.misc import x_dofs_vectorspace
from multi.sampling import create_random_values, correlation_matrix


def test():
    # test / tweak creation of multivariate samples
    # prepare coordinates of DOFs on Γ_out
    num_unit_cells = 3
    resolution = 20
    n = num_unit_cells * resolution
    domain = mesh.create_unit_square(MPI.COMM_SELF, n, n, mesh.CellType.quadrilateral)
    fe = element("P", domain.basix_cell(), 1, shape=(2,))
    V = fem.functionspace(domain, fe)
    everywhere = lambda x: np.full(x[0].shape, True, dtype=bool)
    dofs_gamma = get_boundary_dofs(V, everywhere)
    x_dofs = x_dofs_vectorspace(V)
    points = x_dofs[dofs_gamma]

    ndofs = V.dofmap.bs * V.dofmap.index_map.size_global
    num_cells = domain.topology.index_map(2).size_global
    print(f"""Summary
      DOFs: {ndofs}
      Cells: {num_cells}""")

    # correlation length
    xmin = np.amin(domain.geometry.x, axis=0)
    xmax = np.amax(domain.geometry.x, axis=0)
    L_corr = 10 * np.linalg.norm(xmax - xmin).item()

    num_samples = 0
    while True:
        sigma = correlation_matrix(points, L_corr)
        print(f"Build Sigma of shape {sigma.shape} for {L_corr=}.")
        # Σ = D.dot(csr_array(sigma)).dot(D) # covariance
        λ_max = eigsh(sigma, k=1, which="LM", return_eigenvectors=False)
        rtol = 1e-2
        eigvals = eigh(sigma, eigvals_only=True, driver='evx', subset_by_value=[λ_max.item() * rtol, np.inf])
        num_eigvals = eigvals.size
        print(f"Found {num_eigvals=}.\n")

        mean = np.zeros(sigma.shape[0])
        _ = create_random_values((num_eigvals, sigma.shape[0]), distribution='multivariate_normal', mean=mean, cov=sigma, method='eigh')

        inc = num_eigvals - num_samples
        L_corr /= 2
        if inc > 0:
            num_samples += inc
            print(f"Current {num_samples=}")
        else:
            break
    print(f"Total {num_samples=}")


if __name__ == "__main__":
    test()
