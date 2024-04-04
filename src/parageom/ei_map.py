"""empirical interpolation of the transformation operator"""

from dolfinx import fem
from pymor.parameters.base import ParameterSpace
import basix
import ufl


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

    # x_μ = x_p + d
    # F = Grad(x_μ)
    x_parent = ufl.SpatialCoordinate(aux.problem.domain.grid)
    F = ufl.grad(x_parent + d)
    det_F = ufl.det(F)
    inv_F = ufl.inv(F)
    inv_FT = ufl.transpose(inv_F)
    i, k, j = ufl.indices(3)
    operator_ufl_expr = inv_F[i, k] * inv_FT[k, j] * det_F

    basix_celltype = getattr(basix.CellType, aux.problem.domain.grid.topology.cell_type.name)
    q_degree = 2 # TODO define as BeamData.q_degree
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    operator_expr = fem.Expression(operator_ufl_expr, q_points)

    # TODO
    # Q = QuadratureSpace
    # operator_fun = fem.Function(Q)

    for mu in training_set:
        aux.solve(d, mu)
        d.x.scatter_forward()
        # TODO evaluate expression
        operator_expr.eval(cells, operator_fun.x.array.reshape(cells.size, -1))
        operator_fun.x.scatter_forward()

        # TODO append current operator_fun to list or similar

    # TODO wrap list of operator_fun as pymor VectorArray
    # TODO run ei_greedy

    # compute transformation displacement for each μ.
    # form transformation operator (UFL expr) and interpolate into Qspace for each μ.
    # Run ei greedy on data from step 3.
    pass


if __name__ == "__main__":
    main()
