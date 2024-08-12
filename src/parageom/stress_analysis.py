"""stress analysis"""

import basix
import ufl
import numpy as np
import dolfinx as df
from multi.materials import LinearElasticMaterial


def principal_stress_2d(u: df.fem.Function, q_degree: int, mat: LinearElasticMaterial, values: np.ndarray):
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

    # UFL and fem.Expression
    σ = mat.sigma(u) # type: ignore
    sigma_voigt = ufl.as_vector([σ[0, 0], σ[1, 1], σ[2, 2], σ[0, 1]])
    stress_expr = df.fem.Expression(sigma_voigt, q_points)

    # Compute Cauchy stress
    stress_expr.eval(mesh, entities=cells, values=values)

    # FIXME
    # is this formula correct for the plane strain case, i.e. when s_zz
    # is non-zero?
    def compute_principal_components(f):
        values = f.reshape(cells.size, 4, 4)
        fxx = values[:, :, 0]
        fyy = values[:, :, 1]
        fxy = values[:, :, 3]
        fmin = (fxx+fyy) / 2 - np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        fmax = (fxx+fyy) / 2 + np.sqrt(((fxx-fyy)/2)**2 + fxy**2)
        return fmin, fmax

    # Compute principal stress
    smin, smax = compute_principal_components(values)
    return smin, smax
