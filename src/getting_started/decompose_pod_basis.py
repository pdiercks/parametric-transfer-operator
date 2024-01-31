from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.io import gmshio
from basix.ufl import element
import numpy as np

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt

from multi.bcs import BoundaryDataFactory
from multi.extension import extend
from multi.misc import locate_dofs, x_dofs_vectorspace
from multi.interpolation import make_mapping
from multi.domain import RectangularSubdomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinElaSubProblem
from multi.basis_construction import compute_phi


def main(args):
    """decompose pod basis into coarse and fine scale parts"""
    from .tasks import beam
    gdim = 2
    domain, _, _ = gmshio.read_from_msh(beam.unit_cell_grid.as_posix(), MPI.COMM_WORLD, gdim=gdim)
    omega = RectangularSubdomain(12, domain)
    omega.create_coarse_grid(1)
    omega.create_boundary_grids()

    # ### FE spaces
    degree = beam.fe_deg
    fe = element("P", domain.basix_cell(), degree, shape=(gdim,))
    V = fem.functionspace(omega.grid, fe)
    source = FenicsxVectorSpace(V)

    E = beam.youngs_modulus
    NU = beam.poisson_ratio
    mat = LinearElasticMaterial(gdim, E, NU, plane_stress=False)

    # ### Problem on unit cell domain
    problem = LinElaSubProblem(omega, V, phases=(mat,))
    problem.setup_coarse_space()
    problem.setup_edge_spaces()
    problem.create_map_from_V_to_L()

    # ### Full POD basis
    pod_basis_data = np.load(beam.loc_pod_modes(args.distribution, args.configuration))
    pod_basis = source.from_numpy(pod_basis_data)
    viz = FenicsxVisualizer(source)
    viz.visualize(pod_basis, filename=beam.pod_modes_bp(args.distribution, args.configuration).as_posix())

    # ### Coarse scale basis
    # FIXME: geom_deg is now 2, therefore cannot use full x here
    vertices = omega.coarse_grid.topology.connectivity(2, 0).links(0)
    x_vertices = mesh.compute_midpoints(omega.coarse_grid, 0, vertices)
    phi_vectors = compute_phi(problem, x_vertices)
    phi = source.make_array(phi_vectors)

    # NOTE 16.01.24: looking at the full POD modes in ParaView
    # 1st 20 modes are good, but now the modes get oscillatory
    # and also in the middle and top & bottom the function is basically zero
    # I wonder why those are not removed by the algorithm?
    # Maybe, I have to think more thoroughly about the relation of the target tolerance in the
    # range finder and the POD tolerance.

    # After decomposition into coarse and fine scale part
    # and running gram schmidt on the fine scale edge functions
    # there is a reduction from 68 to ~40 modes
    # I think this should be good, but I would like those oscillatory modes
    # to be not present in the POD basis from the beginning.

    # Is it the POD rtol or ttol of rrf?
    # Would need to check basis for fixed T and see if oscillatory
    # functions occurr even for rather large ttol.

    # The oscillatory effects could be simply introduced by
    # the discontinuous Young's moduli.

    # ### subtract coarse scale part
    xdofs = x_dofs_vectorspace(V)
    node_dofs = locate_dofs(xdofs, x_vertices)
    coarse_dofs = pod_basis.dofs(node_dofs)
    u_coarse = phi.lincomb(coarse_dofs)
    u_fine = pod_basis - u_coarse
    zero_dofs = u_fine.dofs(node_dofs)
    assert np.allclose(zero_dofs, np.zeros_like(zero_dofs))

    # ### Restriction of fine scale part to edges
    bc_factory = BoundaryDataFactory(problem.domain.grid, problem.V)
    edges = set(["left", "bottom", "right", "top"])
    zero_function = fem.Function(problem.V)
    zero_function.x.array[:] = 0.
    boundary_data = list()

    # ### use Gram Schmidt to define edge mode sets
    # bottom-top
    # left-right
    map_top_to_bottom = make_mapping(problem.edge_spaces["fine"]["bottom"], problem.edge_spaces["fine"]["top"])
    map_right_to_left = make_mapping(problem.edge_spaces["fine"]["left"], problem.edge_spaces["fine"]["right"])
    
    left = u_fine.dofs(problem.V_to_L['left'])
    right = u_fine.dofs(problem.V_to_L['right'][map_right_to_left])
    bottom = u_fine.dofs(problem.V_to_L['bottom'])
    top = u_fine.dofs(problem.V_to_L['top'][map_top_to_bottom])
    LR = NumpyVectorSpace(left.shape[-1])
    BT = NumpyVectorSpace(bottom.shape[-1])

    snapshots_left_right = LR.empty()
    snapshots_left_right.append(LR.from_numpy(left))
    snapshots_left_right.append(LR.from_numpy(right))
    modes_left_right = gram_schmidt(snapshots_left_right)

    snapshots_bottom_top = BT.empty()
    snapshots_bottom_top.append(BT.from_numpy(bottom))
    snapshots_bottom_top.append(BT.from_numpy(top))
    modes_bottom_top = gram_schmidt(snapshots_bottom_top)

    # from multi.debug import plot_modes
    # L = problem.edge_spaces['fine'].get('bottom')
    # mask = np.s_[7:8]
    # bottom_modes = modes_bottom_top.to_numpy()
    # plot_modes(L, 'b', bottom_modes, "x", mask)

    mask = {} # mask used to write edge sets separately
    start = 0
    end = 0
    for edge, dofs in problem.V_to_L.items():
        if edge in ("bottom", "top"):
            modes = modes_bottom_top.to_numpy()
        else:
            modes = modes_left_right.to_numpy()
        end += modes.shape[-1]
        mask[edge] = np.s_[start : end]
        start += modes.shape[-1]
        # create BCs for extension of each mode
        zero_boundaries = list(edges.difference([edge]))
        for mode in modes:
            bc = []
            g = bc_factory.create_function_values(mode, dofs)
            bc.append({"value": g, "boundary": problem.domain.str_to_marker(edge),
                       "method": "geometrical"})
            for boundary in zero_boundaries:
                bc.append({"value": zero_function, "boundary": problem.domain.str_to_marker(boundary),
                           "method": "geometrical"})
            assert len(bc) == 4
            boundary_data.append(bc)

    petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            }
    extensions = extend(problem, boundary_data=boundary_data, petsc_options=petsc_options)
    U = source.make_array(extensions)
    viz.visualize(U, filename=beam.fine_scale_modes_bp(args.distribution, args.configuration).as_posix())
    uf_arrays = {}
    for edge, view in mask.items():
        uf_arrays[edge] = U[view].to_numpy()

    # ### write coarse scale basis and fine scale basis as numpy array
    # to be used in the online phase
    # data needs to be written for compatibility with multi.io.BasesLoader
    # FIXME: BasesLoader expects filename basis_{cell_index}.npz
    # Workaround: maybe do not use BasesLoader for this particular example?
    # BasesLoader is designed for case that each coarse grid cell has different
    # set of basis functions

    # TODO: simply write npz file for each configuration
    # BasesLoader config has to be defined to load these files
    # for particular cells
    np.savez(beam.local_basis_npz(args.distribution, args.configuration).as_posix(),
             phi=phi.to_numpy(), **uf_arrays)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("distribution", type=str, help="The distribution used in the range approximation.")
    parser.add_argument("configuration", type=str, help="The type of oversampling problem.", choices=("inner", "left", "right"))
    args = parser.parse_args(sys.argv[1:])
    main(args)
