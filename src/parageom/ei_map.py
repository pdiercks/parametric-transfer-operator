"""empirical interpolation of the transformation operator"""

from dolfinx import fem
from pymor.parameters.base import ParameterSpace
import basix
import ufl

from pymor.bindings.fenicsx import FenicsxVectorSpace
from pymor.algorithms.ei import ei_greedy


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # ### Auxiliary problem
    param = example.parameters["subdomain"]
    aux = discretize_auxiliary_problem(example.parent_unit_cell.as_posix(), example.geom_deg, param)

    # ### Define training set (Parameters({"R": 1}))
    mu_range = example.mu_range
    parameter_space = ParameterSpace(param, mu_range)
    training_set = parameter_space.sample_randomly(10)

    # transformation displacement
    d = fem.Function(aux.problem.V, name="d")

    # ### Deformation gradient of the geometry transformation
    # x_μ = x_p + d
    # F = Grad(x_μ) = I + Grad(d)
    domain = aux.problem.domain
    gdim = domain.gdim
    F = ufl.Identity(gdim) + ufl.grad(d)

    # ### Transformation operator Γ (Gamma)
    det_F = ufl.det(F)
    C = F.T * F
    inv_C = ufl.inv(C)
    op_ufl_expr = inv_C * det_F

    basix_celltype = getattr(basix.CellType, domain.grid.topology.cell_type.name)
    q_degree = 2 # TODO define as BeamData.q_degree
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    op_expr = fem.Expression(op_ufl_expr, q_points)

    # transformation operator is tensor of rank 2 and symmetric
    # if interpolation (see line 59) is used, the value shape must match
    # the shape of the fem.Expression
    qe = basix.ufl.quadrature_element(domain.grid.topology.cell_name(), value_shape=(2, 2), degree=q_degree)
    Q = fem.functionspace(domain.grid, qe)
    op_fun = fem.Function(Q, name="Γ")

    # num_qpoints = Q.dofmap.index_map.size_local # on process
    # dim_qspace = Q.dofmap.index_map.size_global * Q.dofmap.bs

    source = FenicsxVectorSpace(Q)
    snap_vecs = []

    for mu in training_set:
        aux.solve(d, mu)
        d.x.scatter_forward()
        op_fun.interpolate(op_expr)
        snap_vecs.append(op_fun.vector.copy())

    snapshots = source.make_array(snap_vecs)
    # TODO use proper norm
    error_norm = None # uses euclidean norm
    atol = 1e-3
    rtol = None
    interp_dofs, collateral_basis, ei_data = ei_greedy(snapshots, error_norm=error_norm, atol=atol, rtol=rtol, copy=True)

    print(f"Number of basis functions: {len(collateral_basis)}")
    # for atol=1e-3, I get 7 basis functions

    # TODO output
    # interp_dofs
    # collaterial_basis
    # ei_data['errors']
    # ei_data['triangularity_errors']


if __name__ == "__main__":
    main()
