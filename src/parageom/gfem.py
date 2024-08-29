import sys
import pathlib

from mpi4py import MPI
import dolfinx as df
import numpy as np

from multi.io import read_mesh
from multi.domain import StructuredQuadGrid, Domain, RectangularDomain
from multi.solver import build_nullspace

from pymor.algorithms.gram_schmidt import gram_schmidt
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults

# GFEM revisited
# I now have a set of basis functions for each of the 11 target subdomains
# for each of those I have also a mesh

# - Get coarse grid.
# - Load all target subdomains and all bases (or one at a time and destroy?)
# - Loop over cells, for vertex in cell ...
# - NO TRANSLATION SHIT hopefully

# map from vertex to target subdomain
VERT_TO_OMEGA_IN = np.array([0, 1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], dtype=np.int32)


def main(args):
    from .tasks import example
    from .osp_v2 import oversampling_config_factory

    stem = pathlib.Path(__file__).stem  # gfem
    logfilename = example.log_gfem(args.nreal, args.cell).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    if args.debug:
        loglevel = 10 # debug
    else:
        loglevel = 20 # info
    logger = getLogger(stem, level=loglevel)

    # ### Coarse grid partition
    coarse_grid_path = example.coarse_grid("global")
    coarse_domain = read_mesh(coarse_grid_path, MPI.COMM_WORLD, cell_tags=None, kwargs={"gdim": example.gdim})[0]
    struct_grid_gl = StructuredQuadGrid(coarse_domain)

    # Get vertices of current cell
    vertices = struct_grid_gl.get_entities(0, args.cell)
    lower_left_vertex = vertices[:1]
    dx_unit_cell = struct_grid_gl.get_entity_coordinates(0, lower_left_vertex)

    # determine relevant transfer problems based on cell vertices
    transfer_problems = set([])
    for vert in vertices:
        transfer_problems.add(VERT_TO_OMEGA_IN[vert])

    # read mesh for unit cell and translate
    unit_cell_mesh = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})[0]
    unit_cell_domain = Domain(unit_cell_mesh)
    unit_cell_domain.translate(dx_unit_cell)

    breakpoint()
    # ### Function spaces
    value_shape = (example.gdim,)
    X = df.fem.functionspace(struct_grid_gl.grid, ("P", 1, value_shape))  # global coarse space
    V = df.fem.functionspace(unit_cell_domain.grid, ("P", example.fe_deg, value_shape))  # fine space, unit cell level

    # Interpolation data from global coarse grid to unit cell
    coarse_to_unit_cell = df.fem.create_nonmatching_meshes_interpolation_data(
        V.mesh, V.element, X.mesh, padding=1e-10
    )

    # ### Data Structures target subdomain Ω_in
    omega_in = {}
    V_in = {}
    omega_in_to_unit_cell = {} # interpolation data
    bases = {}
    osp_configs = {}
    for k in transfer_problems:
        path_omega_in = example.path_omega_in(k)
        omega_in[k] = RectangularDomain(read_mesh(path_omega_in, MPI.COMM_WORLD, kwargs={"gdim": example.gdim})[0])
        V_in[k] = df.fem.functionspace(omega_in[k].grid, V.ufl_element())
        # Interpolation data from target subdomain to unit cell
        omega_in_to_unit_cell[k] = df.fem.create_nonmatching_meshes_interpolation_data(
            V.mesh, V.element, V_in[k].mesh, padding=1e-10
        )

        # TODO: read local bases
        bases[k] = np.load(example.hapod_modes_npy(args.nreal, k))
        osp_configs[k] = oversampling_config_factory(k)

    # map from vertex to "enrichment"
    # should be determined from kernel
    # Example: cell 0, vertices 0, 1, 2, 3
    # 0, 2 --> k = 0
    # 1, 3 --> k = 1
    enrichments = 0

    def enrich_with_constant(rb, x=False, y=False):
        dim = rb.shape[1]
        xmode = np.ones((1, dim), dtype=np.float64)
        xmode[:, 1::2 ] *= 0.
        ymode = np.ones((1, dim), dtype=np.float64)
        ymode[:, ::2] *= 0.
        basis = rb.copy()
        if y:
            basis = np.vstack([ymode, basis])
        if x:
            basis = np.vstack([xmode, basis])

        return basis

    bases["left_enriched_y"] = enrich_with_constant(bases["left"], y=True)
    # bases["left_enriched_xy"] = enrich_with_constant(bases["left"], x=True, y=True)
    bases["inner_enriched_xy"] = enrich_with_constant(bases["inner"], x=True, y=True)
    # bases["right_enriched_xy"] = enrich_with_constant(bases["right"], x=True, y=True)
    bases["right_enriched_x"] = enrich_with_constant(bases["right"], x=True)

    for k, v in bases.items():
        bases_length[k] = len(v)

    Phi = df.fem.Function(X, name="Phi")  # coarse scale hat functions
    phi = df.fem.Function(V, name="phi")  # hat functions on the fine grid
    xi_in = df.fem.Function(V_in, name="xi_in")  # basis functions on target subdomain
    xi = df.fem.Function(V, name="xi")  # basis function on unit cell grid
    psi = df.fem.Function(V, name="psi")  # GFEM function, psi=phi*xi

    source = FenicsxVectorSpace(V)

    # source_in = FenicsxVectorSpace(V_in)
    # record maximum number of modes per vertex for each cell
    max_modes_per_vertex = []

    modes_per_vertex = []
    for vertex in vertices:
        count_modes_per_vertex = 0
        config = vertex_to_config[vertex]
        basis = bases[vertex_to_basis[vertex]]

        # Translate oversampling domain Ω
        dx_omega = global_quad_grid.get_entity_coordinates(
            0, np.array([vertex], dtype=np.int32)
        )
        # Only translate in x-direction for this particular problem
        dx_omega[:, 1:] = np.zeros_like(dx_omega[:, 1:])
        logger.debug(f"{dx_omega=}")
        if vertex in left_boundary:
            dx_omega = np.array([[a, 0, 0]], dtype=np.float64)
        if vertex in right_boundary:
            dx_omega = np.array([[4 * a, 0, 0]], dtype=np.float64)
        omega[config].translate(dx_omega)
        logger.debug(f"{config=}, \tomega.xmin={omega[config].xmin}")

        # Translate target subdomain Ω_in
        dx_omega_in = dx_omega
        omega_in.translate(dx_omega_in)


        for kk, mode in enumerate(basis):
            # Fill in values for basis
            xi_in.x.petsc_vec.zeroEntries()  # type: ignore
            xi_in.x.petsc_vec.array[:] = mode  # type: ignore
            xi_in.x.scatter_forward()  # type: ignore

            # Interpolate basis function to unit cell grid
            xi.x.petsc_vec.zeroEntries()  # type: ignore
            xi.interpolate(xi_in, nmm_interpolation_data=target_to_unit_cell)  # type: ignore
            xi.x.scatter_forward()  # type: ignore

            # Fill values for hat function on coarse grid
            Phi.x.petsc_vec.zeroEntries()  # type: ignore
            for b in range(X.dofmap.index_map_bs):
                dof = vertex * X.dofmap.index_map_bs + b
                Phi.x.petsc_vec.array[dof] = 1.0  # type: ignore
                Phi.x.scatter_forward()  # type: ignore

            # Interpolate hat function to unit cell grid
            phi.x.petsc_vec.zeroEntries()  # type: ignore
            phi.interpolate(Phi, nmm_interpolation_data=coarse_to_unit_cell)  # type: ignore
            phi.x.scatter_forward()  # type: ignore

            psi.x.petsc_vec.zeroEntries()  # type: ignore
            psi.x.petsc_vec.pointwiseMult(phi.x.petsc_vec, xi.x.petsc_vec)  # type: ignore
            psi.x.scatter_forward()  # type: ignore

            gfem.append(psi.x.petsc_vec.copy())  # type: ignore
            count_modes_per_vertex += 1

        modes_per_vertex.append(count_modes_per_vertex)
        logger.info(
            f"Computed {count_modes_per_vertex} GFEM functions for vertex {vertex} (cell {cell})."
        )
        # reverse translation
        omega[config].translate(-dx_omega)
        omega_in.translate(-dx_omega_in)

        assert len(modes_per_vertex) == 4
        max_modes_per_vertex.append(modes_per_vertex)

        logger.info(f"Computed {len(gfem)} GFEM functions for cell {cell}.")

        # ### Write local gfem basis for cell
        G = source.make_array(gfem)  # type: ignore
        outstream = example.local_basis_npy(
            args.nreal, args.method, args.distribution, cell
        )
        np.save(outstream, G.to_numpy())

        if args.debug:
            outstream_xdmf = outstream.with_suffix(".xdmf")
            viz = FenicsxVisualizer(G.space)
            viz.visualize(G, filename=outstream_xdmf.as_posix())

        # reverse translation
        unit_cell.translate(-dx_unit_cell)

    # end of loop over cells
    max_modes_per_verts = np.array(max_modes_per_vertex, dtype=np.int32)
    assert max_modes_per_verts.shape == (global_quad_grid.num_cells, 4)
    np.save(
        example.local_basis_dofs_per_vert(args.nreal, args.method, args.distribution),
        max_modes_per_verts,
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Construct GFEM functions from local bases."
    )
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("cell", type=int, help="The cell for which to construct GFEM functions.")
    parser.add_argument("--debug", action='store_true', help="Run in debug mode.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
