import pathlib
from mpi4py import MPI
import dolfinx as df
import numpy as np
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid
from multi.misc import x_dofs_vectorspace
from pymor.bindings.fenicsx import FenicsxVectorSpace
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
    logger = getLogger(stem, level="DEBUG")

    # ### coarse grid
    domain, _, _ = read_mesh(example.coarse_grid("global"), MPI.COMM_WORLD, kwargs={"gdim": gdim})

    # ### fine grid 
    struct_grid = StructuredQuadGrid(domain)
    unit_cell, _, _ = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": gdim})
    x_unit_cell = unit_cell.geometry.x
    value_shape = (gdim, )
    V = df.fem.functionspace(unit_cell, ("P", example.fe_deg, value_shape))

    distr = args.distribution
    if args.method == "hapod":
        basis_left = np.load(example.hapod_modes_npy(args.nreal, distr, "left"))
        basis_inner = np.load(example.hapod_modes_npy(args.nreal, distr, "inner"))
        basis_right = np.load(example.hapod_modes_npy(args.nreal, distr, "right"))
    elif args.method == "heuristic":
        basis_left = np.load(example.heuristic_modes_npy(args.nreal, distr, "left"))
        basis_inner = np.load(example.heuristic_modes_npy(args.nreal, distr, "inner"))
        basis_right = np.load(example.heuristic_modes_npy(args.nreal, distr, "right"))
    else:
        raise NotImplementedError
    bases = [basis_left, basis_inner, basis_right]
    bases_length = [len(rb) for rb in bases]
    min_basis_length = min(bases_length)
    logger.debug(f"{min_basis_length=}")

    source = FenicsxVectorSpace(V)
    phi = df.fem.Function(V)
    u = df.fem.Function(V)

    W = df.fem.functionspace(domain, ("P", 1, value_shape))
    w = df.fem.Function(W)
    xdofs = x_dofs_vectorspace(W)
    xverts = struct_grid.grid.geometry.x
    assert xverts.shape[0] == 22
    vertex_to_basis = np.ones(xverts.shape[0], dtype=np.int32)
    vertex_to_basis[[0, 2]] = 0 # left basis for vertex 0 and 2
    vertex_to_basis[[20, 21]] = 2 # right basis for vertex 0 and 2

    vertex_to_owning_cell = np.ones(xverts.shape[0], dtype=np.int32) * 99
    for cell in range(1, 9):
        verts = struct_grid.get_entities(0, cell)
        for v in verts:
            vertex_to_owning_cell[v] = cell
    vertex_to_owning_cell[[0, 2]] = 0 # left boundary
    vertex_to_owning_cell[[20, 21]] = 9 # left boundary

    w_vec = w.x.petsc_vec # hat function in coarse space
    u_vec = u.x.petsc_vec # store basis functions
    phi_vec = phi.x.petsc_vec # store hat function
    xi = df.fem.Function(V) # store product of phi and u
    xi_vec = xi.x.petsc_vec

    for cell in range(struct_grid.num_cells):
        num_gfem_dofs = 0
        gfem = []
        # hats = []

        vertices = struct_grid.get_entities(0, cell)
        x_vertices = struct_grid.get_entity_coordinates(0, vertices)
        dx = x_vertices[0]
        dx = np.around(dx, decimals=3)
        x_unit_cell += dx

        logger.debug(f"{cell=}")
        logger.debug(f"{dx=}")
        logger.debug(f"{vertices=}")

        # compute interpolation data after unit cell domain was translated!
        interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
                V.mesh, V.element, W.mesh, padding=1e-9)

        ldofs = W.dofmap.cell_dofs(cell)
        for ld, vert in zip(ldofs, vertices):
            w_vec.zeroEntries()
            for b in range(W.dofmap.index_map_bs):
                dof = ld * W.dofmap.index_map_bs + b
                assert np.allclose(xdofs[dof], xverts[vert])

                w_vec.array[dof] = 1.0
            w.x.scatter_forward()

            phi_vec.zeroEntries()
            phi.interpolate(w, nmm_interpolation_data=interp_data)
            phi.x.scatter_forward()

            # FIXME
            # the interpolation of w into phi is not leading to
            # very accurate values
            # this yields slightly incompatible GFEM functions
            # its a difference in the third decimal but still
            # this theoretically leads to gaps/overlaps in the
            # global approximation

            logger.debug(f"{vert=}")
            logger.debug(f"{vertex_to_basis[vert]=}")
            basis = bases[vertex_to_basis[vert]]
            sign = 1.0
            if vertex_to_owning_cell[vert] != cell:
                sign = -1.0
                # only the sign is not enough
                # need to know whether its the x- or y-component
                # that needs to be mirrored
                # how would you determine this in a general setting?
                # for the beam I know that only the x-component may
                # has to be mirrored
            for mode in basis[:min_basis_length]:
                u_vec.zeroEntries()
                u_vec.array[::2] = sign * mode.flatten()[::2] # x-component
                u_vec.array[1::2] = mode.flatten()[1::2] # y-component
                u.x.scatter_forward()

                xi_vec.zeroEntries()
                xi_vec.pointwiseMult(phi_vec, u_vec)
                xi.x.scatter_forward()

                gfem.append(xi_vec.copy())
                num_gfem_dofs += 1

        assert len(gfem) == num_gfem_dofs
        logger.debug(f"{cell=}\t{num_gfem_dofs=}")

        G = source.make_array(gfem) # type: ignore
        outpath = example.local_basis_npy(args.nreal, args.method, distr, cell)
        np.save(outpath.as_posix(), G.to_numpy())

        x_unit_cell -= dx


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Construct global approximation via GFEM.")
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("method", type=str, help="Method that was used to construct local bases.")
    parser.add_argument("distribution", type=str, help="Distribution used for random sampling.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
