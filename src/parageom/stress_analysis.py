"""stress analysis"""

import basix
import ufl
import numpy as np
import dolfinx as df
from multi.materials import LinearElasticMaterial


def principal_stress_2d(u: df.fem.Function, q_degree: int, mat: LinearElasticMaterial):
    """Computes principal Cauchy stress.

    Args:
        u: The displacement field.
        q_degree: The quadrature degree.
        mat: The linear elastic material.

    """
    assert not mat.plane_stress

    # Mesh and cells
    mesh = u.function_space.mesh
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    # Quadrature space and Function for Cauchy stress
    basix_celltype = getattr(basix.CellType, mesh.topology.cell_type.name)
    q_points, _ = basix.make_quadrature(basix_celltype, q_degree)
    QVe = basix.ufl.quadrature_element(basix_celltype, value_shape=(4,), scheme="default", degree=q_degree)
    QV = df.fem.functionspace(mesh, QVe)
    stress = df.fem.Function(QV)

    # UFL and fem.Expression
    σ = mat.sigma(u)
    sigma_voigt = ufl.as_vector([σ[0, 0], σ[1, 1], σ[2, 2], σ[0, 1]])
    stress_expr = df.fem.Expression(sigma_voigt, q_points)

    # Compute Cauchy stress
    stress_expr.eval(mesh, entities=cells, values=stress.x.array.reshape(cells.size, -1))

    # Compute principal stress
    # FIXME
    # check if the equation (source) is correct for 2d, but plane strain problems where s_zz
    # is non-zero
    s_values = stress.x.array.reshape(cells.size, 4, 4)
    s_xx = s_values[:, :, 0]
    s_yy = s_values[:, :, 1]
    s_xy = s_values[:, :, 3]
    s_max = (s_xx + s_yy) / 2 + np.sqrt(((s_xx - s_yy) / 2) ** 2 + s_xy**2)
    s_min = (s_xx + s_yy) / 2 - np.sqrt(((s_xx - s_yy) / 2) ** 2 + s_xy**2)

    # FIXME
    # check (how?) that the values are correct?
    # it seems that the load is way too high?
    # I get np.amax(s_max) = 50363 and np.amin(s_min) = -128349 ...
    breakpoint()
