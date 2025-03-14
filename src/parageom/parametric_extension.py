import typing
from collections import namedtuple
from time import perf_counter

import dolfinx as df
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.io import XDMFFile
from mpi4py import MPI
from multi.boundary import plane_at, within_range
from multi.dofmap import QuadrilateralDofLayout
from multi.domain import RectangularDomain, RectangularSubdomain, StructuredQuadGrid
from multi.io import read_mesh
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem, LinElaSubProblem
from multi.product import InnerProduct
from multi.projection import compute_relative_proj_errors
from multi.shapes import NumpyLine, NumpyQuad
from multi.solver import build_nullspace
from petsc4py import PETSc
from pymor.algorithms.basic import project_array
from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.algorithms.pod import pod
from pymor.bindings.fenicsx import FenicsxMatrixOperator, FenicsxVectorSpace, FenicsxVisualizer
from pymor.core.logger import getLogger
from pymor.operators.interface import Operator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import Parameters
from pymor.reductors.basic import extend_basis
from scipy.sparse import csr_array
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import erfinv

from parageom.fom import ParaGeomLinEla


class ExtensionLift(object):
    def __init__(
        self,
        space: FenicsxVectorSpace,
        a_cpp: typing.Union[df.fem.Form, list[typing.Any], typing.Any],
        dofs: npt.NDArray[np.int32],
        boundary: npt.NDArray[np.int32],
    ):
        """Initializes ExtensionLift.

        Args:
            space: The FE space (usually operator range).
            a_cpp: Compiled form of left hand side operator.
            dofs: DOF indices for which values are prescribed via `self.assemble`.
            boundary: All facets comprising the Dirichlet boundary.

        """
        self.range = space
        self._a = a_cpp
        self._x = df.la.create_petsc_vector(space.V.dofmap.index_map, space.V.dofmap.bs)
        tdim = space.V.mesh.topology.dim
        fdim = tdim - 1
        self._g = df.fem.Function(space.V)
        self.dofs = dofs

        # full boundary
        self._dofs = df.fem.locate_dofs_topological(space.V, fdim, boundary)
        self._bcs = [df.fem.dirichletbc(self._g, self._dofs)]

    def _update_dirichlet_data(self, values):
        self._g.x.petsc_vec.zeroEntries()  # type: ignore
        self._g.x.array[self.dofs] = values  # type: ignore
        self._g.x.scatter_forward()  # type: ignore

    def assemble(self, values):
        r = []
        for dofs in values:
            self._update_dirichlet_data(dofs)
            self._x.zeroEntries()
            dolfinx.fem.petsc.apply_lifting(self._x, [self._a], bcs=[self._bcs])  # type: ignore
            self._x.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
            dolfinx.fem.petsc.set_bc(self._x, self._bcs)
            r.append(self._x.copy())

        return self.range.make_array(r)


def get_oversampling_config():
    # index of transfer problem with 3x1 unit cells
    k = 1
    cells_omega = np.array([0, 1, 2], dtype=np.int32)
    cells_omega_in = np.array([1], dtype=np.int32)
    kernel = (0, 1, 2)

    # define left based on smallest cell integer
    unit_length = 1.0
    y_max = 1.0
    left_most_cell = np.amin(cells_omega)
    x_left = float(left_most_cell * unit_length)
    right_most_cell = np.amax(cells_omega)
    x_right = float((right_most_cell + 1) * unit_length)
    # define right based on largest cell integer
    tol = 1e-4
    left = within_range([x_left, 0.0 + tol, 0.0], [x_left, y_max - tol, 0.0])
    right = within_range([x_right, 0.0 + tol, 0.0], [x_right, y_max - tol, 0.0])

    def gamma_out_inner(x):
        return np.logical_or(left(x), right(x))

    gamma_d = None
    gamma_n = None

    OSPConfig = namedtuple(
        'OSPConfig', ['k', 'cells_omega', 'cells_omega_in', 'kernel', 'gamma_out', 'gamma_d', 'gamma_n']
    )
    return OSPConfig(k, cells_omega, cells_omega_in, kernel, gamma_out_inner, gamma_d, gamma_n)


def discretize_transfer_problem(example, coarse_omega, omega, omega_in, osp_config):
    from parageom.auxiliary_problem import GlobalAuxiliaryProblem
    from parageom.locmor import DirichletLift, ParametricTransferProblem
    from parageom.matrix_based_operator import BCGeom, FenicsxMatrixBasedOperator

    cells_omega = osp_config.cells_omega

    # ### Function Spaces
    V = df.fem.functionspace(omega.grid, ('P', example.geom_deg, (example.gdim,)))
    V_in = df.fem.functionspace(omega_in.grid, V.ufl_element())
    source_V_in = FenicsxVectorSpace(V_in)

    # ### Auxiliary problem defined on oversampling domain Omega
    # locate interfaces for definition of auxiliary problem
    left_most_cell = np.amin(cells_omega)
    unit_length = 1.0
    x_min = float(left_most_cell * unit_length)

    interface_locators = []
    for i in range(1, cells_omega.size):
        x_coord = float(x_min + i)
        interface_locators.append(plane_at(x_coord, 'x'))
    # make check
    for marker in interface_locators:
        entities = df.mesh.locate_entities(V.mesh, V.mesh.topology.dim - 1, marker)
        assert entities.size == example.num_intervals

    aux_tags = list(range(15, 15 + cells_omega.size))
    assert len(aux_tags) == cells_omega.size
    assert len(interface_locators) == cells_omega.size - 1
    emod = df.fem.Constant(omega.grid, df.default_scalar_type(1.0))
    nu = df.fem.Constant(omega.grid, df.default_scalar_type(0.25))
    mat = LinearElasticMaterial(example.gdim, E=emod, NU=nu, plane_stress=example.plane_stress)
    problem = LinearElasticityProblem(omega, V, phases=mat)
    params = Parameters({'R': cells_omega.size})
    auxiliary_problem = GlobalAuxiliaryProblem(
        problem, aux_tags, params, coarse_omega, interface_locators=interface_locators
    )
    d_trafo = df.fem.Function(V, name='d_trafo')

    # ### Dirichlet BCs (operator, range product)
    bcs_op = []  # BCs for lhs operator of transfer problem, space V
    zero = df.default_scalar_type(0.0)
    fix_u = df.fem.Constant(V.mesh, (zero,) * example.gdim)
    bc_gamma_out = BCGeom(fix_u, osp_config.gamma_out, V)
    bcs_op.append(bc_gamma_out)
    bcs_op = tuple(bcs_op)

    # ### Discretize left hand side - FenicsxMatrixBasedOperator
    matparam = {
        'gdim': example.gdim,
        'E': example.youngs_modulus,
        'NU': example.poisson_ratio,
        'plane_stress': example.plane_stress,
    }
    parageom = ParaGeomLinEla(
        omega,
        V,
        d=d_trafo,  # type: ignore
        matparam=matparam,
    )

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary_problem.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    # operator for left hand side on full oversampling domain
    operator = FenicsxMatrixBasedOperator(parageom.form_lhs, params, param_setter=param_setter, bcs=bcs_op)

    # ### Discretize right hand side - DirichletLift
    entities_gamma_out = df.mesh.locate_entities_boundary(V.mesh, V.mesh.topology.dim - 1, osp_config.gamma_out)
    expected_num_facets_gamma_out = (example.num_intervals - 2, 2 * (example.num_intervals - 2))
    assert entities_gamma_out.size in expected_num_facets_gamma_out
    rhs = DirichletLift(operator.range, operator.compiled_form, entities_gamma_out)  # type: ignore

    def l2(V):
        """Form for source product."""
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        return ufl.inner(u, v) * ufl.dx  # type: ignore

    # ### Source product operator
    l2_cpp = df.fem.form(l2(V))
    pmat_source = dolfinx.fem.petsc.create_matrix(l2_cpp)  # type: ignore
    pmat_source.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(pmat_source, l2_cpp, bcs=[])
    pmat_source.assemble()
    source_mat = csr_array(pmat_source.getValuesCSR()[::-1])  # type: ignore
    source_product = NumpyMatrixOperator(source_mat[rhs.dofs, :][:, rhs.dofs], name='l2')

    # ### Range Product
    range_mat = LinearElasticMaterial(**matparam)
    linela_target = LinearElasticityProblem(omega_in, V_in, phases=range_mat)
    a_cpp = df.fem.form(linela_target.form_lhs)
    range_mat = dolfinx.fem.petsc.create_matrix(a_cpp)  # type: ignore
    range_mat.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(range_mat, a_cpp, bcs=[])
    range_mat.assemble()
    range_product = FenicsxMatrixOperator(range_mat, V_in, V_in, name='energy')

    # ### Rigid body modes
    kernel_set = osp_config.kernel
    ns_vecs = build_nullspace(V_in, gdim=example.gdim)
    assert len(ns_vecs) == 3
    rigid_body_modes = []

    kernel = None
    if len(kernel_set) > 0:
        for j in kernel_set:
            dolfinx.fem.petsc.set_bc(ns_vecs[j], [])
            rigid_body_modes.append(ns_vecs[j])
        kernel = source_V_in.make_array(rigid_body_modes)  # type: ignore
        gram_schmidt(kernel, product=None, copy=False)
        assert np.allclose(kernel.gramian(), np.eye(len(kernel)))
    assert kernel is not None

    # #### Transfer Problem
    transfer = ParametricTransferProblem(
        operator,
        rhs,
        source_V_in,
        source_product=source_product,
        range_product=range_product,
        kernel=kernel,
        padding=1e-8,
    )
    return transfer


def adaptive_edge_rrf_normal(
    logger,
    transfer_problem,
    active_edges,
    target_subdomain,
    source_product: typing.Optional[Operator] = None,
    range_product: typing.Optional[str] = None,
    error_tol: float = 1e-4,
    failure_tolerance: float = 1e-15,
    num_testvecs: int = 20,
    lambda_min=None,
    **sampling_options,
):
    tp = transfer_problem
    distribution = 'normal'

    assert source_product is None or isinstance(source_product, Operator)
    range_product = range_product or 'h1'

    if source_product is None:
        lambda_min = 1
    elif lambda_min is None:

        def mv(v):
            return source_product.apply(source_product.source.from_numpy(v)).to_numpy()  # type: ignore

        def mvinv(v):
            return source_product.apply_inverse(
                source_product.range.from_numpy(v)  # type: ignore
            ).to_numpy()

        L = LinearOperator(
            (source_product.source.dim, source_product.range.dim),  # type: ignore
            matvec=mv,  # type: ignore
        )
        Linv = LinearOperator(
            (source_product.range.dim, source_product.source.dim),  # type: ignore
            matvec=mvinv,  # type: ignore
        )
        lambda_min = eigsh(L, sigma=0, which='LM', return_eigenvectors=False, k=1, OPinv=Linv)[0]

    # ### test set
    R = tp.generate_random_boundary_data(count=num_testvecs, distribution=distribution)
    M = tp.solve(R)

    dof_layout = QuadrilateralDofLayout()
    edge_index_map = dof_layout.local_edge_index_map

    # ### initialize data structures
    test_set = {}
    range_spaces = {}
    range_products = {}
    pod_bases = {}
    maxnorm = np.array([], dtype=float)
    edges = np.array([], dtype=str)
    coarse_basis = {}
    # the dofs for vertices on the boundary of the edge
    edge_boundary_dofs = {}

    start = perf_counter()
    for i in range(dof_layout.num_entities[1]):
        edge = edge_index_map[i]
        edges = np.append(edges, edge)

        edge_mesh = target_subdomain.domain.fine_edge_grid[edge]
        edge_space = target_subdomain.edge_spaces[edge]
        range_spaces[edge] = FenicsxVectorSpace(edge_space)

        # ### create dirichletbc for range product
        facet_dim = edge_mesh.topology.dim - 1
        vertices = df.mesh.locate_entities_boundary(
            edge_mesh, facet_dim, lambda x: np.full(x[0].shape, True, dtype=bool)
        )
        _dofs_ = df.fem.locate_dofs_topological(edge_space, facet_dim, vertices)
        gdim = target_subdomain.domain.grid.geometry.dim
        range_bc = df.fem.dirichletbc(np.array((0,) * gdim, dtype=df.default_scalar_type), _dofs_, edge_space)
        edge_boundary_dofs[edge] = range_bc._cpp_object.dof_indices()[0]

        # ### range product
        inner_product = InnerProduct(edge_space, range_product, bcs=(range_bc,))
        range_product_op = FenicsxMatrixOperator(inner_product.assemble_matrix(), edge_space, edge_space)
        range_products[edge] = range_product_op

        # ### compute coarse scale edge basis
        nodes = df.mesh.compute_midpoints(edge_mesh, facet_dim, vertices)
        nodes = np.around(nodes, decimals=3)

        component = 0
        if edge in ('left', 'right'):
            component = 1

        line_element = NumpyLine(nodes[:, component])
        shape_funcs = line_element.interpolate(edge_space, component)
        N = range_spaces[edge].from_numpy(shape_funcs)
        coarse_basis[edge] = N

        # ### edge test sets
        dofs = target_subdomain.V_to_L[edge]
        test_set[edge] = range_spaces[edge].from_numpy(M.dofs(dofs))
        # subtract coarse scale part
        test_cvals = test_set[edge].dofs(edge_boundary_dofs[edge])
        test_set[edge] -= N.lincomb(test_cvals)
        assert np.isclose(np.sum(test_set[edge].dofs(edge_boundary_dofs[edge])), 1e-9)

        # ### initialize maxnorm
        if edge in active_edges:
            maxnorm = np.append(maxnorm, np.inf)
            # ### pod bases
            pod_bases[edge] = range_spaces[edge].empty()
        else:
            maxnorm = np.append(maxnorm, 0.0)

    end = perf_counter()
    logger.debug(f'Preparing stuff took t={end-start}s.')

    # NOTE tp.source is the full space, while the source product
    # is of lower dimension
    num_source_dofs = len(tp.rhs.dofs)
    testfail = np.array([failure_tolerance / min(num_source_dofs, space.dim) for space in range_spaces.values()])
    testlimit = np.sqrt(2.0 * lambda_min) * erfinv(testfail ** (1.0 / num_testvecs)) * error_tol

    logger.info(f'{lambda_min=}')
    logger.info(f'{testlimit=}')

    num_solves = 0
    while np.any(maxnorm > testlimit):
        v = tp.generate_random_boundary_data(1, distribution, **sampling_options)

        U = tp.solve(v)
        num_solves += 1

        target_edges = edges[maxnorm > testlimit]
        for edge in target_edges:
            B = pod_bases[edge]
            edge_space = range_spaces[edge]
            # restrict the training sample to the edge
            Udofs = edge_space.from_numpy(U.dofs(target_subdomain.V_to_L[edge]))  # FIXME
            coarse_values = Udofs.dofs(edge_boundary_dofs[edge])
            U_fine = Udofs - coarse_basis[edge].lincomb(coarse_values)

            # extend pod basis
            extend_basis(U_fine, B, product=range_products[edge], method='gram_schmidt')

            # orthonormalize test set wrt pod basis
            M = test_set[edge]
            M -= B.lincomb(B.inner(M, range_products[edge]).T)

            norm = M.norm(range_products[edge])
            maxnorm[edge_index_map[edge]] = np.max(norm)

        logger.debug(f'{maxnorm=}')

    return pod_bases, range_products, num_solves


def generate_test_data(example, transfer, num_samples, num_testvecs):
    """Returns test data."""
    from parageom.lhs import sample_lhs

    assert transfer.kernel is None

    ps = transfer.operator.parameters.space(example.mu_range)
    testing_set = sample_lhs(ps, name='R', samples=num_samples, criterion='center', random_state=7111987)

    test_data = transfer.range.empty(reserve=num_samples * num_testvecs)
    for mu in testing_set:
        transfer.assemble_operator(mu)
        R = transfer.generate_random_boundary_data(num_testvecs, distribution='normal')
        U = transfer.solve(R)
        test_data.append(U)

    return test_data


def main(args):
    from parageom.auxiliary_problem import discretize_auxiliary_problem
    from parageom.matrix_based_operator import BCTopo, FenicsxMatrixBasedOperator
    from parageom.tasks import example

    logger = getLogger('parametric_oversampling', level=10)

    osp_config = get_oversampling_config()

    # coarse grid of oversampling domain
    coarse_grid_path = example.path_omega_coarse(osp_config.k)
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': example.gdim})[0]
    struct_grid = StructuredQuadGrid(coarse_domain)

    # ### Fine grid partition of omega
    path_omega = example.path_omega(osp_config.k)
    with XDMFFile(MPI.COMM_WORLD, path_omega.as_posix(), 'r') as xdmf:
        omega_mesh = xdmf.read_mesh()
        omega_ct = xdmf.read_meshtags(omega_mesh, name='Cell tags')
        omega_ft = xdmf.read_meshtags(omega_mesh, name='mesh_tags')
    omega = RectangularDomain(omega_mesh, cell_tags=omega_ct, facet_tags=omega_ft)

    # ### Fine grid partition of omega in
    path_omega_in = example.parent_unit_cell
    omega_in, omega_in_ct, omega_in_ft = read_mesh(
        path_omega_in, MPI.COMM_WORLD, cell_tags=None, kwargs={'gdim': example.gdim}
    )
    omega_in = RectangularSubdomain(99, omega_in, omega_in_ct, omega_in_ft)
    omega_in.translate(np.array([[1.0, 0.0, 0.0]], dtype=np.float64))

    # TODO meshes for edges of target subdomain
    omega_in.create_coarse_grid(1)
    omega_in.create_boundary_grids()
    # TODO spaces for edges of target subdomain
    mat = LinearElasticMaterial(
        gdim=example.gdim, E=example.youngs_modulus, NU=example.poisson_ratio, plane_stress=example.plane_stress
    )
    W = df.fem.functionspace(omega_in.grid, ('P', example.fe_deg, (example.gdim,)))
    subproblem = LinElaSubProblem(omega_in, W, phases=mat)
    subproblem.setup_coarse_space()
    subproblem.setup_edge_spaces()
    subproblem.create_map_from_V_to_L()

    TargetSubdomainWrapper = namedtuple('TargetSubdomainWrapper', ['domain', 'edge_spaces', 'V_to_L'])
    target_subdomain = TargetSubdomainWrapper(omega_in, subproblem.edge_spaces['fine'], subproblem.V_to_L)

    transfer = discretize_transfer_problem(example, struct_grid, omega, omega_in, osp_config)
    # fext = transfer.operator.range.zeros(1)

    mu_ref = transfer.operator.parameters.parse([0.2 for _ in range(3)])
    transfer.assemble_operator(mu_ref)

    active_edges = set(['bottom', 'left', 'right', 'top'])
    pod_bases, range_products, num_solves = adaptive_edge_rrf_normal(
        logger,
        transfer,
        active_edges,
        target_subdomain,
        source_product=transfer.source_product,
        range_product='h1',
        error_tol=1e-4,
        failure_tolerance=1e-14,
        num_testvecs=20,
        lambda_min=None,
    )

    # ### Discretize left hand side of Extension Problem - FenicsxMatrixBasedOperator
    # first define auxiliary problem on unit cell (target subdomain)
    ftags = {'bottom': 11, 'left': 12, 'right': 13, 'top': 14, 'interface': 15}
    paramdim = {'R': 1}
    auxiliary = discretize_auxiliary_problem(example, omega_in, ftags, paramdim)
    d_trafo = df.fem.Function(W)

    matparam = {
        'gdim': example.gdim,
        'E': example.youngs_modulus,
        'NU': example.poisson_ratio,
        'plane_stress': example.plane_stress,
    }
    parageom = ParaGeomLinEla(omega_in, W, d=d_trafo, matparam=matparam)

    def param_setter(mu):
        d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
        auxiliary.solve(d_trafo, mu)  # type: ignore
        d_trafo.x.scatter_forward()  # type: ignore

    facets_left = omega_in.facet_tags.find(ftags['left'])
    facets_right = omega_in.facet_tags.find(ftags['right'])
    facets_bottom = omega_in.facet_tags.find(ftags['bottom'])
    facets_top = omega_in.facet_tags.find(ftags['top'])
    boundary = np.hstack([facets_left, facets_right, facets_bottom, facets_top])

    # operator for left hand side on target subdomain Ω_in
    bcs_op = []  # BCs for lhs operator of extension problem
    zero = df.default_scalar_type(0.0)
    fix_u = df.fem.Constant(W.mesh, (zero,) * example.gdim)
    tdim = omega_in.tdim
    fdim = tdim - 1
    bc_boundary = BCTopo(fix_u, boundary, fdim, W)
    bcs_op.append(bc_boundary)
    extop = FenicsxMatrixBasedOperator(parageom.form_lhs, paramdim, param_setter=param_setter, bcs=tuple(bcs_op))

    # definition of the training set
    # use uniform sampling since parameter space is small P=[R_min, R_max]
    ntrain = 20
    ps = extop.parameters.space(example.mu_range)
    training_set = ps.sample_uniformly(ntrain)

    # rhs operator (ExtensionLift) for each edge
    rhs_op = {}
    snapshots = {}
    num_modes = {}
    for key, dofs in target_subdomain.V_to_L.items():
        rhs_op[key] = ExtensionLift(extop.range, extop.compiled_form, dofs, boundary)
        num_modes[key] = len(pod_bases[key])
        snapshots[key] = extop.source.empty(reserve=ntrain * num_modes[key])

    # we need for each mode, the extension for every mu in the training set
    # however, computing all extensions after assemble(mu) once is faster

    # how are the snapshots collected best?
    # - I will have ntrain snapshots for each mode for each edge
    # - e.g. ntrain * num_modes = 20 * 13 = 260 for the left edge

    # for edge in ["bottom", "left", "right", "top"]:
    logger.debug('Starting Extension for Edge Modes')
    edges = ['bottom', 'left', 'right', 'top']
    for edge in edges:
        for mu in training_set:
            lhs = extop.assemble(mu)
            modes = pod_bases[edge].to_numpy()  # all modes for `edge`
            R = rhs_op[edge].assemble(modes)
            U = lhs.apply_inverse(R)  # all extensions for `edge`
            snapshots[edge].append(U)

    # snapshots correpsonding to mode 0 have indices
    # 0, 13, 26, ..., 247
    # num_modes = num_modes["left"]
    # view_mode_0 = np.arange(0, num_modes * (ntrain-1) + 1, num_modes, dtype=np.int32)
    # view_mode_1 = view_mode_0 + 1 # etc.

    # test first single pod
    # TODO range product for target subdomain space W

    pod_modes = {}
    for edge in edges:
        pod_modes[edge] = extop.source.empty()
        for i in range(num_modes[edge]):
            # print(f"{i+1}-th mode")
            view = np.arange(i, num_modes[edge] * (ntrain - 1) + 1 + i, num_modes[edge], dtype=np.int32)
            # print(view)
            rb, sv = pod(snapshots[edge][view], product=None, rtol=1e-6)
            # print(f"Adding {len(rb_left)} POD modes")
            pod_modes[edge].append(rb)

        viz = FenicsxVisualizer(pod_modes[edge].space)
        viz.visualize(pod_modes[edge], filename=f'output/extended_pod_modes_{edge}.xdmf')
        print(f'Computed {len(pod_modes[edge])} for {edge=}.')

    # TODO parametric extension of coarse scale basis functions
    # re-use `extop`
    # define new ExtensionLift

    boundary_dofs = np.hstack(
        [
            df.fem.locate_dofs_topological(extop.source.V.sub(0), fdim, boundary),
            df.fem.locate_dofs_topological(extop.source.V.sub(1), fdim, boundary),
        ]
    )
    ext_lift = ExtensionLift(extop.range, extop.compiled_form, boundary_dofs, boundary)

    # TODO this can be moved inside the loop above?
    xmin = omega_in.xmin
    xmax = omega_in.xmax
    nodes = np.array(
        [
            [xmin[0], xmin[1], xmin[2]],
            [xmax[0], xmin[1], xmin[2]],
            [xmin[0], xmax[1], xmin[2]],
            [xmax[0], xmax[1], xmin[2]],
        ],
        dtype=np.float64,
    )
    quad = NumpyQuad(nodes)
    shapes = quad.interpolate(extop.source.V)
    modes = shapes[:, boundary_dofs]
    R = ext_lift.assemble(modes)

    coarse_snapshots = extop.source.empty()
    for mu in training_set:
        lhs = extop.assemble(mu)
        U = lhs.apply_inverse(R)
        coarse_snapshots.append(U)

    # FIXME
    # you have to split x and y components
    # 4 nodes, 2 modes each
    # 160 snapshots, e.g. 20 snapshots for node i (0, ..., 7) and component j (0, 1)
    # viz = FenicsxVisualizer(extop.source)

    logger.debug('Starting Extension for Coarse Scale Modes')
    coarse_modes = {}
    for i in range(8):
        view = np.arange(i, 8 * (ntrain - 1) + 1 + i, 8, dtype=np.int32)
        coarse_pod_basis, coarse_svals = pod(coarse_snapshots[view], product=None, rtol=1e-6)
        coarse_modes[i] = coarse_pod_basis

    logger.debug('Generating test data')
    ntest = 100
    ntestvecs = 2
    transfer.kernel = None
    test_data = generate_test_data(example, transfer, ntest, ntestvecs)

    basis = extop.source.empty()
    for i in range(8):
        basis.append(coarse_modes[i])
    for j in range(80):
        for edge in edges:
            basis.append(pod_modes[edge][j])

    U = extop.source.from_numpy(test_data.to_numpy())  # FIXME
    relerr = compute_relative_proj_errors(U, basis, product=None, orthonormal=False)
    # this does not seem to be working
    # maybe compute projection and plot via paraview?
    U_proj = project_array(U, basis, product=None, orthonormal=False)
    E = U - U_proj
    viz = FenicsxVisualizer(basis.space)
    viz.visualize(U, filename='output/test_data.xdmf')
    viz.visualize(U_proj, filename='output/test_data_projection.xdmf')
    viz.visualize(E, filename='output/test_data_error.xdmf')
    breakpoint()

    # relerr[-1] = 0.00188
    # TODO use energy product?

    # Does this mean we could use this?
    # The error decay is quite modest, but this is usually the case because of the
    # decomposition.
    # I think I would see similar modest decay if I would use the GFEM functions here.


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        'Proof of concept example for non-parametric oversampling with parametric extension'
    )
    args = parser.parse_args(sys.argv[1:])
    main(args)
