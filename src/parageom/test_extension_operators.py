from typing import Union, Any

import numpy as np
import numpy.typing as npt
import dolfinx as df
import dolfinx.fem.petsc
from parageom.tasks import example
from parageom.auxiliary_problem import discretize_auxiliary_problem
from parageom.fom import ParaGeomLinEla
from parageom.matrix_based_operator import FenicsxMatrixBasedOperator, BCTopo
from mpi4py import MPI
from multi.io import read_mesh
from multi.domain import RectangularSubdomain
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from scipy.sparse import csr_array
from petsc4py import PETSc

class DirichletLift(object):
    def __init__(
        self,
        space: FenicsxVectorSpace,
        a_cpp: Union[df.fem.Form, list[Any], Any],
        facets: npt.NDArray[np.int32],
    ):
        self.range = space
        self._a = a_cpp
        self._x = df.la.create_petsc_vector(space.V.dofmap.index_map, space.V.dofmap.bs)  # type: ignore
        tdim = space.V.mesh.topology.dim  # type: ignore
        fdim = tdim - 1
        self._dofs = df.fem.locate_dofs_topological(space.V, fdim, facets)  # type: ignore
        self._g = df.fem.Function(space.V)  # type: ignore
        self._bcs = [df.fem.dirichletbc(self._g, self._dofs)]  # type: ignore
        self.dofs = self._bcs[0]._cpp_object.dof_indices()[0]

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


# ### Fine grid partition of omega in
path_omega_in = example.parent_unit_cell
omega_in, omega_in_ct, omega_in_ft = read_mesh(path_omega_in, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": example.gdim})
omega_in = RectangularSubdomain(99, omega_in, omega_in_ct, omega_in_ft)
W = df.fem.functionspace(omega_in.grid, ("P", example.fe_deg, (example.gdim,)))

ftags = {"bottom": 11, "left": 12, "right": 13, "top": 14, "interface": 15}
paramdim = {"R": 1}
auxiliary = discretize_auxiliary_problem(example, omega_in, ftags, paramdim)
d_trafo = df.fem.Function(W)

matparam = {"gdim": example.gdim, "E": example.youngs_modulus, "NU": example.poisson_ratio, "plane_stress": example.plane_stress}
parageom = ParaGeomLinEla(omega_in, W, d=d_trafo, matparam=matparam)

def param_setter(mu):
    d_trafo.x.petsc_vec.zeroEntries()  # type: ignore
    auxiliary.solve(d_trafo, mu)  # type: ignore
    d_trafo.x.scatter_forward()  # type: ignore

def boundary_omega_in(x):
    return np.full(x[0].shape, True, dtype=bool)

zero_fun = df.fem.Constant(W.mesh, (df.default_scalar_type(0.0),)*2)
parageom.add_dirichlet_bc(zero_fun, omega_in.str_to_marker("bottom"), method="geometrical")
parageom.add_dirichlet_bc(zero_fun, omega_in.str_to_marker("right"), method="geometrical")
parageom.add_dirichlet_bc(zero_fun, omega_in.str_to_marker("top"), method="geometrical")

one_fun = df.fem.Constant(W.mesh, (df.default_scalar_type(1.0),)*2)
parageom.add_dirichlet_bc(one_fun, omega_in.str_to_marker("left"), method="geometrical")

mu_ref = auxiliary.parameters.parse([0.1])
param_setter(mu_ref)
parageom.setup_solver()
u = parageom.solve()

rhs_array_ref = parageom.b.array[:]

source = FenicsxVectorSpace(W)
U = source.make_array([u.x.petsc_vec])
viz = FenicsxVisualizer(source)
viz.visualize(U, filename="extU_ref.xdmf")

# operator for left hand side on full oversampling domain
bcs_op = [] # BCs for lhs operator of extension problem
zero = df.default_scalar_type(0.0)
fix_u = df.fem.Constant(W.mesh, (zero,) * example.gdim)
tdim = omega_in.tdim
fdim = tdim - 1
# facets_boundary_omega_in = df.mesh.locate_entities_boundary(omega_in.grid, fdim, boundary_omega_in)
facets_left = omega_in.facet_tags.find(ftags["left"])
facets_right = omega_in.facet_tags.find(ftags["right"])
facets_bottom = omega_in.facet_tags.find(ftags["bottom"])
facets_top = omega_in.facet_tags.find(ftags["top"])
bcs_op.append(BCTopo(fix_u, facets_left, fdim, W))
bcs_op.append(BCTopo(fix_u, facets_right, fdim, W))
bcs_op.append(BCTopo(fix_u, facets_bottom, fdim, W))
bcs_op.append(BCTopo(fix_u, facets_top, fdim, W))
extop = FenicsxMatrixBasedOperator(
    parageom.form_lhs, paramdim, param_setter=param_setter, bcs=tuple(bcs_op)
)

# define rhs operator (DirichletLift) for each edge
facets_left = omega_in.facet_tags.find(ftags["left"])
rhs_left = DirichletLift(extop.range, extop.compiled_form, facets_left)

all_facets = np.hstack([facets_left, facets_right, facets_bottom, facets_top])
other_rhs = DirichletLift(extop.range, extop.compiled_form, all_facets)

# TODO
# understand why `all_facets` need to be passed when the whole boundary is constrained
# 
# if e.g. left=g, right=0 (bottom & top no bc), then it suffices to pass `facets_left`
# to DirichletLift

data = np.zeros((1, 200-8))
data[:, :50] = 1.
other_R = other_rhs.assemble(data)

mu_ref = auxiliary.parameters.parse([0.1])
lhs = extop.assemble(mu_ref)

R = rhs_left.assemble(np.ones((1, 50)))
U = lhs.apply_inverse(R)
viz.visualize(U, filename="extU_other.xdmf")

A_ref = csr_array(parageom.A.getValuesCSR()[::-1])
A = csr_array(lhs.matrix.getValuesCSR()[::-1])
lhs_equal = np.allclose(A_ref.todense(), A.todense())
rhs_equal = np.allclose(rhs_array_ref, R.to_numpy().flatten())
other_rhs_equal = np.allclose(rhs_array_ref, other_R.to_numpy().flatten())
breakpoint()
if lhs_equal:
    print("lhs equal")
if rhs_equal:
    print("rhs equal")
