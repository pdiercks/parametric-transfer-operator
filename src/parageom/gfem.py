import sys
import pathlib
import tempfile

from mpi4py import MPI
import dolfinx as df
import numpy as np

from multi.preprocessing import create_rectangle
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid, RectangularDomain
from multi.boundary import plane_at

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults


def main(args):
    from .tasks import example

    gdim = example.gdim

    stem = pathlib.Path(__file__).stem  # gfem
    logfilename = example.log_gfem(
        args.nreal, args.method, args.distribution
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    if args.debug:
        loglevel = 10 # debug
    else:
        loglevel = 20 # info
    logger = getLogger(stem, level=loglevel)

    # ### Grids
    # global quadrilateral grid for GFEM construction
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        xmin = ymin = 0.0
        xmax = 5.0
        ymax = 1.0
        num_cells = [5, 1]
        create_rectangle(
            xmin,
            xmax,
            ymin,
            ymax,
            num_cells=num_cells,
            recombine=True,
            out_file=tf.name,
        )
        global_coarse_grid = read_mesh(
            pathlib.Path(tf.name), MPI.COMM_WORLD, kwargs={"gdim": gdim}
        )[0]

    global_quad_grid = StructuredQuadGrid(global_coarse_grid)
    vertex_to_basis = {
        0: "left_enriched_y",
        1: "left_enriched_xy",
        2: "left_enriched_y",
        3: "left_enriched_xy",
        4: "inner_enriched_xy",
        5: "inner_enriched_xy",
        6: "inner_enriched_xy",
        7: "inner_enriched_xy",
        8: "right_enriched_xy",
        9: "right_enriched_xy",
        10: "right_enriched_x",
        11: "right_enriched_xy",
            }
    vertex_to_config = {
        0: "left",
        1: "left",
        2: "left",
        3: "left",
        4: "inner",
        5: "inner",
        6: "inner",
        7: "inner",
        8: "right",
        9: "right",
        10: "right",
        11: "right",
    }
    left_boundary = global_quad_grid.locate_entities_boundary(0, plane_at(xmin, "x"))
    right_boundary = global_quad_grid.locate_entities_boundary(0, plane_at(xmax, "x"))

    # oversampling domain Ω for each configuration
    # see parageom/preprocessing.py for xmin
    # left, xmin = [0, 0, 0]
    # inner, xmin = [3a, 0, 0]
    # right, xmin = [7a, 0, 0]
    omega = {}
    for config in example.configurations:
        filepath = example.parent_domain(config)
        domain = read_mesh(filepath, MPI.COMM_WORLD, kwargs={"gdim": gdim})[0]
        omega[config] = RectangularDomain(domain)

    # target subdomain Ω_in, xmin = [0, 0, 0]
    domain_in = read_mesh(
        example.target_subdomain, MPI.COMM_WORLD, kwargs={"gdim": gdim}
    )[0]
    omega_in = RectangularDomain(domain_in)

    # translate Ω and Ω_in such that x=[0, 0, 0] ∈ Γ, where Γ is the interface
    # shared by the two coarse grid cells of Ω_in
    a = example.unit_length
    omega_in.translate(np.array([[-a, 0.0, 0.0]]))  # xmin = [-a, 0, 0]
    omega["left"].translate(np.array([[-a, 0.0, 0.0]]))  # xmin = [-a, 0, 0]
    omega["inner"].translate(np.array([[-5 * a, 0.0, 0.0]]))  # xmin = [-2a, 0, 0]
    omega["right"].translate(np.array([[-9 * a, 0.0, 0.0]]))  # xmin = [-2a, 0, 0]

    # unit cell grid Ω_i, xmin = [0, 0, 0]
    unit_cell_grid = read_mesh(
        example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": gdim}
    )[0]
    unit_cell = RectangularDomain(unit_cell_grid)

    # ### Function spaces
    value_shape = (gdim,)
    X = df.fem.functionspace(
        global_coarse_grid, ("P", 1, value_shape)
    )  # global coarse space
    V = df.fem.functionspace(
        unit_cell.grid, ("P", example.fe_deg, value_shape)
    )  # fine space, unit cell level
    V_in = df.fem.functionspace(
        omega_in.grid, V.ufl_element()
    )  # fine space, target subdomain Ω_in
    source = FenicsxVectorSpace(V)

    # read local bases
    distr = args.distribution
    bases = {}
    bases_length = {}
    if args.method == "hapod":
        fpath_modes_npy = example.hapod_modes_npy
    elif args.method == "heuristic":
        fpath_modes_npy = example.heuristic_modes_npy
    else:
        raise NotImplementedError
    bases["left"] = np.load(fpath_modes_npy(args.nreal, distr, "left"))
    bases["inner"] = np.load(fpath_modes_npy(args.nreal, distr, "inner"))
    bases["right"] = np.load(fpath_modes_npy(args.nreal, distr, "right"))

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
    bases["left_enriched_xy"] = enrich_with_constant(bases["left"], x=True, y=True)
    bases["inner_enriched_xy"] = enrich_with_constant(bases["inner"], x=True, y=True)
    bases["right_enriched_xy"] = enrich_with_constant(bases["right"], x=True, y=True)
    bases["right_enriched_x"] = enrich_with_constant(bases["right"], x=True)

    for k, v in bases.items():
        bases_length[k] = len(v)

    Phi = df.fem.Function(X, name="Phi")  # coarse scale hat functions
    phi = df.fem.Function(V, name="phi")  # hat functions on the fine grid
    xi_in = df.fem.Function(V_in, name="xi_in")  # basis functions on target subdomain
    xi = df.fem.Function(V, name="xi")  # basis function on unit cell grid
    psi = df.fem.Function(V, name="psi")  # GFEM function, psi=phi*xi

    # record maximum number of modes per vertex for each cell
    max_modes_per_vertex = []

    for cell in range(global_quad_grid.num_cells):
        gfem = []

        # Get vertices of current cell and translate unit cell
        vertices = global_quad_grid.get_entities(0, cell)
        # first vertex is always the lower left vertex
        dx_unit_cell = global_quad_grid.get_entity_coordinates(0, vertices[:1])
        unit_cell.translate(dx_unit_cell)
        logger.debug(f"{cell=}, \t{unit_cell.xmin=}")

        # Create interpolation data after translation of the unit cell
        coarse_to_unit_cell = df.fem.create_nonmatching_meshes_interpolation_data(
            V.mesh, V.element, X.mesh, padding=1e-10
        )

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

            # Create interpolation data after translation of target subdomain
            target_to_unit_cell = df.fem.create_nonmatching_meshes_interpolation_data(
                V.mesh, V.element, V_in.mesh, padding=1e-10
            )

            for mode in basis:
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
        description="Construct global approximation via GFEM."
    )
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument(
        "method", type=str, help="Method that was used to construct local bases."
    )
    parser.add_argument(
        "distribution", type=str, help="Distribution used for random sampling."
    )
    parser.add_argument("--debug", action='store_true', help="Run in debug mode.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
