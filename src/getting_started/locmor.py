from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_array

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Mu, Parameters, ParameterSpace

from multi.boundary import point_at, plane_at, within_range
from multi.domain import RectangularDomain, RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinElaSubProblem, TransferProblem
from .definitions import BeamData, BeamProblem


class COOMatrixOperator(Operator):
    """Wraps COO matrix data as an |Operator|.

    Args:
        data: COO matrix data. See scipy.sparse.coo_array.
        indexptr: Points to end of data for each cell.
        num_cells: Number of cells.
        shape: The shape of the matrix.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray], indexptr: np.ndarray, num_cells: int, shape: Tuple[int, int], solver_options: Optional[dict] = None, name: Optional[str] = None):
        assert all([d.shape == data[0].shape for d in data])
        self.__auto_init(locals()) # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        data, rows, cols = self.data # type: ignore
        indexptr = self.indexptr # type: ignore
        num_cells = self.num_cells # type: ignore

        if mu is not None:
            m = mu.to_numpy()
            data[:indexptr[0]] *= m[0]
            for i in range(1, num_cells):
                data[indexptr[i-1]:indexptr[i]] *= m[i]

        K = coo_array((data, (rows, cols)), shape=self.shape) # type: ignore
        K.eliminate_zeros()
        return NumpyMatrixOperator(K.tocsr(), self.source.id, self.range.id, self.solver_options, self.name + "_assembled")

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)


def discretize_oversampling_problem(example: BeamData, mu: Mu, configuration: str):
    """Returns TransferProblem for fixed parameter Mu.

    Args:
        example: The instance of the example dataclass.
        mu: The parameter value.
        configuration: The type of oversampling problem.

    """

    # use MPI.COMM_SELF for embarrassingly parallel workloads
    with XDMFFile(MPI.COMM_SELF, example.fine_oversampling_grid(configuration).as_posix(), "r") as fh:
        domain = fh.read_mesh(name="Grid")
        cell_tags = fh.read_meshtags(domain, "subdomains")

    omega = RectangularDomain(domain, cell_tags=cell_tags, facet_tags=None)
    omega.create_facet_tags({
        "bottom": int(11), "left": int(12), "right": int(13), "top": int(14)
        })
    tdim = omega.tdim
    gdim = omega.gdim

    # ### Definitions dependent on configuration
    # Topology: Γ_out, Ω_in, Σ_D
    # Dirichlet BCs on Σ_D
    mu_values = mu.to_numpy()
    beamproblem = BeamProblem(example.coarse_grid, example.fine_grid)
    cell_index = beamproblem.config_to_cell(configuration)
    gamma_out = beamproblem.get_gamma_out(cell_index)
    mark_omega_in = beamproblem.get_omega_in(cell_index)
    dirichlet = beamproblem.get_dirichlet(cell_index)
    remove_kernel = beamproblem.get_remove_kernel(cell_index)

    if configuration == "inner":
        assert mu_values.size == 3
        assert omega.facet_tags.find(11).size == example.resolution * 3 # bottom
        assert omega.facet_tags.find(12).size == example.resolution * 1 # left
        assert omega.facet_tags.find(13).size == example.resolution * 1 # right
        assert omega.facet_tags.find(14).size == example.resolution * 3 # top

    elif configuration == "left":
        assert mu_values.size == 2
        assert omega.facet_tags.find(11).size == example.resolution * 2 # bottom
        assert omega.facet_tags.find(12).size == example.resolution * 1 # left
        assert omega.facet_tags.find(13).size == example.resolution * 1 # right
        assert omega.facet_tags.find(14).size == example.resolution * 2 # top

    elif configuration == "right":
        assert mu_values.size == 2
        assert omega.facet_tags.find(11).size == example.resolution * 2 # bottom
        assert omega.facet_tags.find(12).size == example.resolution * 1 # left
        assert omega.facet_tags.find(13).size == example.resolution * 1 # right
        assert omega.facet_tags.find(14).size == example.resolution * 2 # top

    else:
        raise NotImplementedError

    # ### Omega in
    cells_omega_in = mesh.locate_entities(domain, tdim, mark_omega_in)
    omega_in, _, _, _ = mesh.create_submesh(domain, tdim, cells_omega_in)
    id_omega_in = 99
    omega_in = RectangularSubdomain(id_omega_in, omega_in)

    # ### FE spaces
    degree = example.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe) # full space
    W = fem.functionspace(omega_in.grid, fe) # range space

    # ### Oversampling problem
    E = example.youngs_modulus
    NU = example.poisson_ratio
    materials = tuple([LinearElasticMaterial(gdim, E * mu_i, NU, plane_stress=False) for mu_i in mu_values])
    oversampling_problem = LinearElasticityProblem(omega, V, phases=materials)

    # ### Problem on target subdomain
    if configuration == "inner":
        subproblem = LinElaSubProblem(omega_in, W, phases=(materials[1],))
    elif configuration == "left":
        subproblem = LinElaSubProblem(omega_in, W, phases=(materials[0],))
    elif configuration == "right":
        subproblem = LinElaSubProblem(omega_in, W, phases=(materials[-1],))
    else:
        raise NotImplementedError

    # ### TransferProblem
    transfer = TransferProblem(
            oversampling_problem,
            subproblem,
            gamma_out,
            dirichlet=dirichlet,
            source_product={"product": "l2"},
            range_product={"product": "h1"},
            # remove_kernel=remove_kernel,
            remove_kernel=False, # FIXME: instead of bool, indicate set of rigid body modes to use
            )
    return transfer




if __name__ == "__main__":
    from .tasks import beam
    from multi.misc import x_dofs_vectorspace, locate_dofs
    from pymor.bindings.fenicsx import FenicsxVisualizer
    param = Parameters({"E": 2})
    ps = ParameterSpace(param, (1., 2.))
    mu = ps.sample_randomly(1)[0]
    T = discretize_oversampling_problem(beam, mu, "left")
    v = T.generate_random_boundary_data(1, distribution='normal')
    breakpoint()
    U = T.solve(v)
    xdofs = x_dofs_vectorspace(T.range.V)
    dofs_origin = locate_dofs(xdofs, np.array([[0, 0, 0]]))
    viz = FenicsxVisualizer(T.range)
    viz.visualize(U, filename="./homDirichlet.xdmf")
