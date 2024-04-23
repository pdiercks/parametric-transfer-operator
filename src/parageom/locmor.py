from typing import Tuple, Optional

import numpy as np
from scipy.sparse import coo_array

from mpi4py import MPI
from dolfinx import fem
from dolfinx.fem.petsc import set_bc
from dolfinx.io import gmshio
from dolfinx.io.utils import XDMFFile
from basix.ufl import element

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.parameters.base import Parameters, ParameterSpace

from multi.domain import RectangularDomain, RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinElaSubProblem, TransferProblem
from multi.product import InnerProduct
from multi.solver import build_nullspace
from multi.io import read_mesh
from .definitions import BeamData, BeamProblem


class COOMatrixOperator(Operator):
    """Wraps COO matrix data as an |Operator|.

    Args:
        data: COO matrix data. See scipy.sparse.coo_array.
        indexptr: Points to end of data for each cell.
        num_cells: Number of cells.
        shape: The shape of the matrix.
        parameters: The |Parameters| the operator depends on.
        solver_options: Solver options.
        name: The name of the operator.

    """

    linear = True

    def __init__(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray], indexptr: np.ndarray, num_cells: int, shape: Tuple[int, int], parameters: Parameters = {}, solver_options: Optional[dict] = None, name: Optional[str] = None):
        assert all([d.shape == data[0].shape for d in data])
        self.__auto_init(locals()) # type: ignore
        self.source = NumpyVectorSpace(shape[1])
        self.range = NumpyVectorSpace(shape[0])
        self._data = data[0].copy()

    def assemble(self, mu=None):
        assert self.parameters.assert_compatible(mu)

        data, rows, cols = self.data # type: ignore
        indexptr = self.indexptr # type: ignore
        num_cells = self.num_cells # type: ignore

        new = self._data
        if self.parametric and mu is not None:
            m = mu.to_numpy()
            new[:indexptr[0]] = data[:indexptr[0]] * m[0]
            for i in range(1, num_cells):
                new[indexptr[i-1]:indexptr[i]] = data[indexptr[i-1]:indexptr[i]] * m[i]

        K = coo_array((new, (rows, cols)), shape=self.shape) # type: ignore
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


def discretize_oversampling_problem(example: BeamData, configuration: str, index: int):
    """Returns TransferProblem for fixed parameter Mu.

    Args:
        example: The instance of the example dataclass.
        configuration: The type of oversampling problem.
        index: The index of the parameter value of the training set.

    """

    # use MPI.COMM_SELF for embarrassingly parallel workloads
    oversamplingdomain_xdmf = example.oversampling_domain(configuration, index).as_posix()
    with XDMFFile(MPI.COMM_SELF, oversamplingdomain_xdmf, "r") as fh:
        domain = fh.read_mesh(name="Grid")

    omega = RectangularDomain(domain, cell_tags=None, facet_tags=None)
    omega.create_facet_tags({
        "bottom": int(11), "left": int(12), "right": int(13), "top": int(14)
        })

    if configuration == "inner":
        assert omega.facet_tags.find(11).size == example.num_intervals * 3 # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1 # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1 # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 3 # top

    elif configuration == "left":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2 # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1 # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1 # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2 # top

    elif configuration == "right":
        assert omega.facet_tags.find(11).size == example.num_intervals * 2 # bottom
        assert omega.facet_tags.find(12).size == example.num_intervals * 1 # left
        assert omega.facet_tags.find(13).size == example.num_intervals * 1 # right
        assert omega.facet_tags.find(14).size == example.num_intervals * 2 # top

    else:
        raise NotImplementedError

    # ### Definitions dependent on configuration
    # Topology: Γ_out, Ω_in, Σ_D
    # Dirichlet BCs on Σ_D

    # BeamProblem is only used to get stuff for transfer problem definition
    # here we do not need the actual physical meshes?
    beamproblem = BeamProblem(example.coarse_grid("global"), example.global_parent_domain)
    cell_index = beamproblem.config_to_cell(configuration)
    gamma_out = beamproblem.get_gamma_out(cell_index)
    dirichlet = beamproblem.get_dirichlet(cell_index)
    kernel_set = beamproblem.get_kernel_set(cell_index)

    # ### Omega in
    gdim = example.gdim
    target_subdomain, _, _ = read_mesh(example.target_subdomain(configuration, index), MPI.COMM_WORLD, gdim=gdim)
    id_omega_in = 99
    omega_in = RectangularSubdomain(id_omega_in, target_subdomain)
    # create coarse grid of target subdomain
    # required for fine scale part computation using coarse FE space
    omega_in.create_coarse_grid(1)
    omega_in.create_boundary_grids()

    # ### FE spaces
    degree = example.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe) # full space
    W = fem.functionspace(omega_in.grid, fe) # range space

    # ### Oversampling problem
    emod = example.youngs_modulus
    nu = example.poisson_ratio
    material = LinearElasticMaterial(gdim, E=emod, NU=nu, plane_stress=False)
    oversampling_problem = LinearElasticityProblem(omega, V, phases=material)

    # ### Problem on target subdomain
    # definition of correct material for consistency
    # however, unless energy inner product is used as inner product for the range
    # space, this should not have influence on the solution
    subproblem = LinElaSubProblem(omega_in, W, phases=material)
    # required for fine scale part computation using coarse FE space
    subproblem.setup_coarse_space()
    subproblem.setup_edge_spaces()
    subproblem.create_map_from_V_to_L()

    # ### Range product operator
    # get homogeneous Dirichlet bcs if present
    bc_hom = []
    if dirichlet is not None:
        subproblem.add_dirichlet_bc(**dirichlet)
        bc_hom = subproblem.get_dirichlet_bcs()

    inner_product = InnerProduct(subproblem.V, example.range_product, bcs=bc_hom)
    pmat = inner_product.assemble_matrix()
    range_product = FenicsxMatrixOperator(pmat, subproblem.V, subproblem.V)

    # ### Rigid body modes
    ns_vecs = build_nullspace(subproblem.V, gdim=omega_in.grid.geometry.dim)
    range_space = FenicsxVectorSpace(subproblem.V)
    rigid_body_modes = []
    for j in kernel_set:
        set_bc(ns_vecs[j], bc_hom)
        rigid_body_modes.append(ns_vecs[j])
    kernel = range_space.make_array(rigid_body_modes)
    gram_schmidt(kernel, product=range_product, copy=False)

    # ### TransferProblem
    oversampling_problem.clear_bcs()
    subproblem.clear_bcs()
    transfer = TransferProblem(
            oversampling_problem,
            subproblem,
            gamma_out,
            dirichlet=dirichlet,
            source_product={"product": "l2", "bcs": ()},
            range_product=range_product,
            kernel=kernel,
            )
    return transfer


if __name__ == "__main__":
    from .tasks import example
    from multi.misc import x_dofs_vectorspace, locate_dofs
    from pymor.bindings.fenicsx import FenicsxVisualizer
    param = Parameters({"E": 2})
    ps = ParameterSpace(param, (1., 2.))
    mu = ps.parameters.parse([1.5 for _ in range(2)])
    configuration = "right"
    # configuration = "left"
    index = 0
    T = discretize_oversampling_problem(example, configuration, index)
    v = T.generate_random_boundary_data(1, distribution='normal')
    v[:, ::2] = 0.1 # set x component to value
    v[:, 1::2] = 0.1 # set y component to value
    U = T.solve(v)

    xdofs = x_dofs_vectorspace(T.range.V)
    if configuration == "left":
        dofs = locate_dofs(xdofs, np.array([[0., 0., 0.]]))
    elif configuration == "right":
        dofs = locate_dofs(xdofs, np.array([[10., 0., 0.]]))
    assert np.allclose(U.dofs(dofs)[:, 1], np.zeros_like(U.dofs(dofs)[:, 1]))

    viz = FenicsxVisualizer(T.range)
    viz.visualize(U, filename="./homDirichlet.bp")
