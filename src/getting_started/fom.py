import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from definitions import Example


def main():
    ex = Example(name="beam")
    fom = discretize_fom(ex)


def discretize_fom(ex):
    """returns FOM as pymor model"""

    # read fine grid from disk
    with XDMFFile(MPI.COMM_WORLD, ex.fine_grid.as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, name="Grid")

    # finite element space
    gdim = domain.ufl_cell().geometric_dimension()
    ve = element("P", domain.basix_cell(), ex.fe_deg, shape=(gdim,))
    V = fem.functionspace(domain, ve)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def strain(x):
        """Assume plane strain for ease of implementation"""
        e = ufl.sym(ufl.grad(x))
        return e

    # UFL weak form
    dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    # we have the same a_q, q=1, 2 for each sudomain id, id=range(10)

    def ass_mat(subdomain_id):
        """assemble matrix"""
        eps = strain(u)
        δeps = strain(v)
        i, j = ufl.indices(2)
        a_1 = eps[i, i] * δeps[j, j] * dx(subdomain_id)
        a_2 = e[i, j] * δeps[i, j] * dx(subdomain_id)


    # assemble matrices

    # parameter functionals (Lame constants)
    # L = E nu / (1 + nu) / (1-2nu)
    # M = E / 2 / (1+nu)

    return None


if __name__ == "__main__":
    main()
