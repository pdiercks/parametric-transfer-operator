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
from pymor.core.logger import getLogger, set_log_levels
from pymor.core.defaults import set_defaults


def main(args):
    from .tasks import example

    gdim = example.gdim

    stem = pathlib.Path(__file__).stem  # gfem
    logfilename = example.log_gfem(
        args.nreal, args.method, args.distribution
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger(stem)
    # TODO
    # consider setting log level via example.log_level
    set_log_levels({"pymor": 30, "pymor.algorithms.gram_schmidt.gram_schmidt": 30, stem: 10})

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
    domain_in = read_mesh(example.target_subdomain, MPI.COMM_WORLD, kwargs={"gdim": gdim})[0]
    omega_in = RectangularDomain(domain_in)

    # translate Ω and Ω_in such that x=[0, 0, 0] ∈ Γ, where Γ is the interface
    # shared by the two coarse grid cells of Ω_in
    a = example.unit_length
    omega_in.translate(np.array([[-a, 0., 0.]])) # xmin = [-a, 0, 0]
    omega["left"].translate(np.array([[-a, 0., 0.]])) # xmin = [-a, 0, 0]
    omega["inner"].translate(np.array([[-5*a, 0., 0.]])) # xmin = [-2a, 0, 0]
    omega["right"].translate(np.array([[-9*a, 0., 0.]])) # xmin = [-2a, 0, 0]

    # unit cell grid Ω_i, xmin = [0, 0, 0]
    unit_cell_grid = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": gdim})[0]
    unit_cell = RectangularDomain(unit_cell_grid)

    # ### Function spaces
    value_shape = (gdim,)
    X = df.fem.functionspace(global_coarse_grid, ("P", 1, value_shape)) # global coarse space
    V = df.fem.functionspace(unit_cell.grid, ("P", example.fe_deg, value_shape)) # fine space, unit cell level
    V_in = df.fem.functionspace(omega_in.grid, V.ufl_element()) # fine space, target subdomain Ω_in
    source = FenicsxVectorSpace(V)
    # visualizer = FenicsxVisualizer(source)

    # read local bases
    distr = args.distribution
    bases = {}
    bases_length = {}
    for config in example.configurations:
        if args.method == "hapod":
            bases[config] = np.load(example.hapod_modes_npy(args.nreal, distr, config))
        elif args.method == "heuristic":
            bases[config] = np.load(example.heuristic_modes_npy(args.nreal, distr, config))
        else:
            raise NotImplementedError
        bases_length[config] = len(bases[config])
    assert len(list(bases.keys())) == 3

    Phi = df.fem.Function(X, name="Phi") # coarse scale hat functions
    phi = df.fem.Function(V, name="phi") # hat functions on the fine grid
    xi_in = df.fem.Function(V_in, name="xi_in") # basis functions on target subdomain
    xi = df.fem.Function(V, name="xi") # basis function on unit cell grid
    psi = df.fem.Function(V, name="psi") # GFEM function, psi=phi*xi

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
                V.mesh,
                V.element,
                X.mesh,
                padding=1e-10)


        for vertex in vertices:
            config = vertex_to_config[vertex]
            basis = bases[config]

            # Translate oversampling domain Ω
            dx_omega = global_quad_grid.get_entity_coordinates(0, np.array([vertex], dtype=np.int32))
            # Only translate in x-direction for this particular problem
            dx_omega[:, 1:] = np.zeros_like(dx_omega[:, 1:]) 
            logger.debug(f"{dx_omega=}")
            if vertex in left_boundary:
                dx_omega = np.array([[a, 0, 0]], dtype=np.float32)
            if vertex in right_boundary:
                dx_omega = np.array([[9*a, 0, 0]], dtype=np.float32)
            omega[config].translate(dx_omega)
            logger.debug(f"{config=}, \tomega.xmin={omega[config].xmin}")

            # Translate target subdomain Ω_in
            dx_omega_in = dx_omega
            omega_in.translate(dx_omega_in)

            # Create interpolation data after translation of target subdomain
            target_to_unit_cell = df.fem.create_nonmatching_meshes_interpolation_data(
                    V.mesh,
                    V.element,
                    V_in.mesh,
                    padding=1e-10)

            # Fill values for hat function on coarse grid
            Phi.x.petsc_vec.zeroEntries()
            for b in range(X.dofmap.index_map_bs):
                dof = vertex * X.dofmap.index_map_bs + b
                Phi.x.petsc_vec.array[dof] = 1.0
            Phi.x.scatter_forward()

            # Interpolate hat function to unit cell grid
            phi.x.petsc_vec.zeroEntries()
            phi.interpolate(Phi, nmm_interpolation_data=coarse_to_unit_cell)
            phi.x.scatter_forward()

            for mode in basis:
                # Fill in values for basis
                xi_in.x.petsc_vec.zeroEntries()
                xi_in.x.petsc_vec.array[:] = mode
                xi_in.x.scatter_forward()

                # Interpolate basis function to unit cell grid
                xi.x.petsc_vec.zeroEntries()
                xi.interpolate(xi_in, nmm_interpolation_data=target_to_unit_cell)
                xi.x.scatter_forward()

                psi.x.petsc_vec.zeroEntries()
                psi.x.petsc_vec.pointwiseMult(phi.x.petsc_vec, xi.x.petsc_vec)
                psi.x.scatter_forward()

                # TODO
                # normalize, such that psi(node) = 1?
                # would have to normalize each component separately
                # not necessary though

                gfem.append(psi.x.petsc_vec.copy())

            logger.info(f"Computed {len(gfem)} GFEM functions for vertex {vertex} (cell {cell}).")
            # reverse translation
            omega[config].translate(-dx_omega)
            omega_in.translate(-dx_omega_in)

        G = source.make_array(gfem)
        outstream = example.local_basis_npy(args.nreal, args.method, args.distribution, cell)
        np.save(outstream, G.to_numpy())
        logger.info(f"Computed {len(gfem)} GFEM functions for cell {cell}.")
        # visualizer.visualize(G, filename=f"gfem_{cell}.xdmf")

        # reverse translation
        unit_cell.translate(-dx_unit_cell)


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
    args = parser.parse_args(sys.argv[1:])
    main(args)
