import pathlib
import tempfile
from mpi4py import MPI
import dolfinx as df
import numpy as np
from multi.preprocessing import create_rectangle
from multi.io import read_mesh
from multi.domain import StructuredQuadGrid
from multi.misc import x_dofs_vectorspace
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxVisualizer


def main():
    from .tasks import example
    unit_length = example.unit_length
    xmin = ymin = 0.
    xmax = 2 * unit_length
    ymax = 1 * unit_length
    gdim = 2

    # ### coarse grid
    domain, _, _ = read_mesh(example.coarse_grid("global"), MPI.COMM_WORLD, kwargs={"gdim": gdim})
    # with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
    #     create_rectangle(xmin, xmax, ymin, ymax, num_cells=(2, 1), recombine=True, out_file=tf.name)
    #     domain, _, _ = read_mesh(pathlib.Path(tf.name), MPI.COMM_WORLD, kwargs={"gdim": gdim})

    # ### fine grid 
    # unit_cell_msh = example.parent_unit_cell.as_posix()
    struct_grid = StructuredQuadGrid(domain)
    # struct_grid.fine_grid_method = [unit_cell_msh]
    # with tempfile.NamedTemporaryFile(suffix=".xdmf") as tf:
    # struct_grid.create_fine_grid(np.arange(struct_grid.num_cells), "mymesh.xdmf", "quad9")
    # xdmf written by meshio has `name` 'Grid'
    # fine_domain = read_mesh(pathlib.Path("mymesh.xdmf"), MPI.COMM_WORLD, kwargs={"name": "Grid"})[0]

    # actually I do not need the fine grid
    # but only the unit cell domain that is translated

    # is there an easy way to get the node for some dof?

    unit_cell, ct, ft = read_mesh(example.parent_unit_cell, MPI.COMM_WORLD, kwargs={"gdim": gdim})
    x_unit_cell = unit_cell.geometry.x
    value_shape = (gdim, )
    V = df.fem.functionspace(unit_cell, ("P", 2, value_shape))
    basis_left = np.load(example.hapod_modes_npy(0, "normal", "left"))
    basis_inner = np.load(example.hapod_modes_npy(0, "normal", "inner"))
    basis_right = np.load(example.hapod_modes_npy(0, "normal", "right"))
    bases = [basis_left, basis_inner, basis_right]
    bases_length = [len(rb) for rb in bases]
    min_basis_length = min(bases_length)
    print(f"{min_basis_length=}")
    source = FenicsxVectorSpace(V)
    phi = df.fem.Function(V)
    viz = FenicsxVisualizer(source)
    u = df.fem.Function(V)

    W = df.fem.functionspace(domain, ("P", 1, value_shape))
    w = df.fem.Function(W)
    # num_dofs = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
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

    breakpoint()

    w_vec = w.x.petsc_vec # hat function in coarse space
    u_vec = u.x.petsc_vec # store basis functions
    phi_vec = phi.x.petsc_vec # store hat function
    xi = df.fem.Function(V) # store product of phi and u
    xi_vec = xi.x.petsc_vec

    for cell in range(struct_grid.num_cells):
        num_gfem_dofs = 0
        gfem = []
        hats = []

        vertices = struct_grid.get_entities(0, cell)
        x_vertices = struct_grid.get_entity_coordinates(0, vertices)
        dx = x_vertices[0]
        dx = np.around(dx, decimals=2)
        x_unit_cell += dx

        print(f"{cell=}")
        print(f"{dx=}")
        print(f"{vertices=}")

        # compute interpolation data after unit cell domain was translated!
        interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
                V.mesh, V.element, W.mesh, padding=1e-14)

        ldofs = W.dofmap.cell_dofs(cell)
        for ld, vert in zip(ldofs, vertices):
            # print(f"{vert=}\t {xverts[vert]=}")
            w_vec.zeroEntries()
            for b in range(W.dofmap.index_map_bs):
                dof = ld * W.dofmap.index_map_bs + b
                # print(f"{dof=}\t {xdofs[dof]=}")
                assert np.allclose(xdofs[dof], xverts[vert])

                # w_vec.zeroEntries()
                w_vec.array[dof] = 1.0
                # w.x.scatter_forward()
            w.x.scatter_forward()

            phi_vec.zeroEntries()
            phi.interpolate(w, nmm_interpolation_data=interp_data)
            phi.x.scatter_forward()

            print("phi max: ", np.amax(phi_vec.array))
            hats.append(phi_vec.copy())

            # FIXME
            # the interpolation of w into phi is not leading to
            # very accurate values
            # this yields slightly incompatible GFEM functions
            # its a difference in the third decimal but still
            # this theoretically leads to gaps/overlaps in the
            # global approximation

            # TODO: need to loop over basis set of the archetype
            print(f"{vert=}")
            print(f"{vertex_to_basis[vert]=}")
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
        assert len(hats) == 4
        print(f"{cell=}\t{num_gfem_dofs=}")
        # G = np.vstack(gfem)
        # GG = source.from_numpy(G)
        # P = np.vstack(hats)
        # PHI = source.make_array(hats)
        GG = source.make_array(gfem)
        viz.visualize(GG, filename=f"gg_{cell}.xdmf")
        # viz.visualize(PHI, filename=f"phi_{cell}.xdmf")
        # breakpoint()
        # w.x.array[i] = 1.0
        # interpolate u into subdomain space to get local hat function
        # also requires to locate the dof and associated node and from the node choose the subdomain
        x_unit_cell -= dx

if __name__ == "__main__":
    main()
