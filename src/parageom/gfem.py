import sys
import pathlib

from mpi4py import MPI
import dolfinx as df
import numpy as np

from multi.io import read_mesh
from multi.domain import StructuredQuadGrid, Domain, RectangularDomain

from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer
from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults

# map from vertex to target subdomain
VERT_TO_OMEGA_IN = np.array([0, 1, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10], dtype=np.int32)


def enrichment_from_kernel(kernel):
    r = {}
    if 0 in kernel:
        r["x"] = True
    if 1 in kernel:
        r["y"] = True
    return r


def main(args):
    from parageom.tasks import example
    from parageom.locmor import oversampling_config_factory

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

    # ### Function spaces
    value_shape = (example.gdim,)
    X = df.fem.functionspace(struct_grid_gl.grid, ("P", 1, value_shape))  # global coarse space
    V = df.fem.functionspace(unit_cell_domain.grid, ("P", example.fe_deg, value_shape))  # fine space, unit cell level

    # Interpolation data from global coarse grid to unit cell
    coarse_to_unit_cell = df.fem.create_nonmatching_meshes_interpolation_data(
        V.mesh, V.element, X.mesh, padding=1e-10
    )

    # ### Discussion Enrichment

    # The only question that remains:
    # If boundary of Ω_in coincides with boundary Sigma_D,
    # do we enrich only by not constrained component (i) or not at all (ii)?

    # I will first go with enriching based on kernel (i)
    # If this still is shit might try (ii) again

    # (iii) enrich vertices on left or right boundary only by RBM missing from kernel
    # and enrich with x and y everywhere else

    # (i) Gives stiff ROM response.
    # (iii) Gives response that "looks" okay. I need to run again with better training data and more modes to be sure. The condition number now is of order 1e+10.

    # Given the observations from (i) & (iii),
    # I think that (ii) should give also way too stiff response.
    # (i) should have been the most consistent theory-wise!!

    # map from vertex to "enrichment"
    # should be determined from kernel
    # Example: cell 1, vertices 1, 4, 3, 5
    # 1, 3 --> k = 1, kernel = (1, 2)    --> enrich x
    # 4, 5 --> k = 2, kernel = (0, 1, 2) --> enrich x and y

    def enrichment_from_option_three(v):
        assert v in list(range(22))

        if v in (0, 2):
            return {"y": True}
        elif v in (20, ):
            return {"x": True}
        else:
            return {"x": True, "y": True}

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

    # ### Data Structures target subdomain Ω_in
    omega_in = {}
    V_in = {}
    xi_in = {} # basis function on target subdomain
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
        xi_in[k] = df.fem.Function(V_in[k], name=f"xi_in_{k:02}")  # basis functions on target subdomain

        osp_configs[k] = oversampling_config_factory(k)
        # enrich = enrichment_from_kernel(osp_configs[k].kernel)
        # bases[k] = enrich_with_constant(np.load(example.hapod_modes_npy(args.nreal, k)), **enrich)

        # first load bases without enrichment
        bases[k] = np.load(example.hapod_modes_npy(args.nreal, k))

        # FIXME
        # in the above version there is only one enrichment possible per k
        # however, always two vertices belong to k and we might want to enrich
        # differently (right boundary)


    Phi = df.fem.Function(X, name="Phi")  # coarse scale hat functions
    phi = df.fem.Function(V, name="phi")  # hat functions on the fine grid
    xi = df.fem.Function(V, name="xi")  # basis function on unit cell grid
    psi = df.fem.Function(V, name="psi")  # GFEM function, psi=phi*xi on unit cell

    modes_per_vertex = []
    gfem = []

    for vertex in vertices:

        count_modes_per_vertex = 0
        k = VERT_TO_OMEGA_IN[vertex]
        enrich = enrichment_from_option_three(vertex)
        blength = len(bases[k])
        # this makes unnecessary copies, but for testing okay
        # FIXME avoid unnecessary copies
        basis = enrich_with_constant(bases[k], **enrich)
        assert np.isclose(len(bases[k]), blength)
        assert len(basis) > blength

        for mode in basis:
            # Fill in values for basis
            xi_in[k].x.petsc_vec.zeroEntries()  # type: ignore
            xi_in[k].x.petsc_vec.array[:] = mode  # type: ignore
            xi_in[k].x.scatter_forward()  # type: ignore

            # Interpolate basis function to unit cell grid
            xi.x.petsc_vec.zeroEntries()  # type: ignore
            xi.interpolate(xi_in[k], nmm_interpolation_data=omega_in_to_unit_cell[k])  # type: ignore
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
            f"Computed {count_modes_per_vertex} GFEM functions for vertex {vertex} (cell {args.cell})."
        )

    assert len(modes_per_vertex) == 4

    logger.info(f"Total of {len(gfem)} GFEM functions for cell {args.cell}.")

    # ### Write local gfem basis for cell
    source = FenicsxVectorSpace(V)
    G = source.make_array(gfem)  # type: ignore
    outstream = example.local_basis_npy(args.nreal, args.cell)
    np.save(outstream, G.to_numpy())

    # Write dofs per vertex for dofmap construction of ROM
    np.save(example.local_basis_dofs_per_vert(args.nreal, args.cell), modes_per_vertex)

    if args.debug:
        outstream_xdmf = outstream.with_suffix(".xdmf")
        viz = FenicsxVisualizer(G.space)
        viz.visualize(G, filename=outstream_xdmf.as_posix())


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
