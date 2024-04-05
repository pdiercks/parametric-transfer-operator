"""empirical interpolation of the transformation operator"""

from typing import Union

from dolfinx import fem
from pymor.parameters.base import ParameterSpace
import basix
import ufl

import numpy as np
from scipy.linalg import solve_triangular
from pymor.bindings.fenicsx import FenicsxVectorSpace
from pymor.algorithms.ei import ei_greedy


def interpolate_transformation_operator(example, test: bool = True, output: Union[str, None] = None):
    from .auxiliary_problem import discretize_auxiliary_problem

    # ### Auxiliary problem
    param = example.parameters["subdomain"]
    aux = discretize_auxiliary_problem(example.parent_unit_cell.as_posix(), example.geom_deg, param)

    # ### Define training set (Parameters({"R": 1}))
    mu_range = example.mu_range
    parameter_space = ParameterSpace(param, mu_range)
    training_set = parameter_space.sample_randomly(100)

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

    print("Computing snapshots of transformation operator for training set of"
          f" size {len(training_set)}.")
    for mu in training_set:
        aux.solve(d, mu)
        d.x.scatter_forward()
        op_fun.interpolate(op_expr)
        op_fun.x.scatter_forward()
        snap_vecs.append(op_fun.vector.copy())

    snapshots = source.make_array(snap_vecs)

    # TODO use proper norm
    # operator: FenicsxMatrixOperator
    # norm = pymor.constructions.induced_norm(operator)
    # FIXME 
    # FeniscxMatrixOperator.matrix would need to have size (dim_qspace, dim_qspace)

    # Is Euclidean norm actually not problematic, because I am measuring an element of
    # a quadrature space?

    error_norm = None # uses euclidean norm
    atol = 1e-3
    rtol = None
    interp_dofs, collateral_basis, ei_data = ei_greedy(snapshots, error_norm=error_norm, atol=atol, rtol=rtol, copy=True)

    print(f"Number of basis functions: {len(collateral_basis)}")

    # make test
    if test:
        interpolation_matrix = collateral_basis.dofs(interp_dofs).T
        validation_set = parameter_space.sample_randomly(20)
        errors = []
        for mu in validation_set:
            # fom
            aux.solve(d, mu)
            d.x.scatter_forward()
            op_fun.interpolate(op_expr)
            op_fun.x.scatter_forward()

            # reduced representation
            # FIXME restricted evaluation is not implemented
            # therefore using full operator here for AU
            AU = op_fun.x.array[interp_dofs]
            coeff = solve_triangular(interpolation_matrix, AU, lower=True, unit_diagonal=True).T
            op_ei = collateral_basis.lincomb(coeff)
            
            # error
            err = op_fun.x.array - op_ei.to_numpy().flatten()
            errors.append(np.linalg.norm(err))

        print(f"Max. error {np.amax(errors)} over validation set of size {len(validation_set)}.")

    # TODO: restricted evaluation
    # use fem.Expression.eval instead of op_fun.interpolate
    # fem.Expression.eval requires cells
    # need to determine cells that contain interp_dofs
    # figure out how to filter out qpoints that are not interp_dofs

    if output is not None:
        print("output not implemented")
        # TODO output
        # interp_dofs
        # collaterial_basis
        # ei_data['errors']
        # ei_data['triangularity_errors']


if __name__ == "__main__":
    from .tasks import example
    interpolate_transformation_operator(example, test=True, output=None)

