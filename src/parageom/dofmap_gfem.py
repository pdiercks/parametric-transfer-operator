"""DofMap for GFEM approximation."""

from typing import Union

import numpy as np
import numpy.typing as npt
from multi.domain import StructuredQuadGrid
from multi.dofmap import QuadrilateralDofLayout


class GFEMDofMap(object):
    def __init__(self, grid: StructuredQuadGrid):
        self.grid = grid
        self.dof_layout = QuadrilateralDofLayout()

        tdim = grid.tdim
        grid.grid.topology.create_connectivity(tdim, 0)
        cell_to_vertices = grid.grid.topology.connectivity(tdim, 0)
        self.c2v = cell_to_vertices
        self.num_cells = grid.num_cells
        self.num_vertices = np.unique(cell_to_vertices.array).size

    def distribute_dofs(self, dofs_per_vert: Union[int, npt.NDArray[np.int32]]):
        """Distributes DOF indices.

        Args:
            dofs_per_vert: Number of DOFs per vertex.

        """

        num_cells = self.num_cells
        layout = self.dof_layout
        dofs_per_edge = 0
        dofs_per_face = 0
        cell_to_vertices = self.c2v
        dmap = {}
        dof_counter = 0

        if isinstance(dofs_per_vert, (int, np.integer)):
            dofs_per_vert = np.ones((num_cells, 4), dtype=np.int32) * dofs_per_vert
        else:
            assert dofs_per_vert.shape == (num_cells, 4)

        for cell_index in range(num_cells):
            layout.set_entity_dofs((dofs_per_vert[cell_index], dofs_per_edge, dofs_per_face))
            entity_dofs = self.dof_layout.get_entity_dofs()

            vertices = cell_to_vertices.links(cell_index)
            for local_ent, entity in enumerate(vertices):
                if entity not in dmap.keys():
                    dmap[entity] = list()
                    vertex_dofs = entity_dofs[0][local_ent]
                    for _ in vertex_dofs:
                        dmap[entity].append(dof_counter)
                        dof_counter += 1

        self._dmap = dmap
        self._n_dofs = dof_counter
        self.dofs_per_vert = dofs_per_vert
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_face = dofs_per_face

    @property
    def num_dofs(self) -> int:
        """Returns total number of DOFs."""
        if not hasattr(self, "_n_dofs"):
            raise AttributeError("You need to distribute DoFs first")
        return self._n_dofs

    def entity_dofs(self, entity: int) -> list[int]:
        """Return all dofs for entity `entity`."""

        if not hasattr(self, "_dmap"):
            raise AttributeError("You need to distribute DoFs first")

        return self._dmap[entity]

    def cell_dofs(self, cell_index: int) -> list[int]:
        """Returns DOFs for given cell."""

        if not hasattr(self, "_dmap"):
            raise AttributeError("You need to distribute DoFs first")

        num_cells = self.num_cells
        assert cell_index in np.arange(num_cells)

        cell_dofs = []
        cell_to_vertices = self.c2v
        vertices = cell_to_vertices.links(cell_index)
        for vert in vertices:
            cell_dofs += self._dmap[vert]
        return cell_dofs


def parageom_dof_distribution_factory(n: int, nmax: dict[str, int]) -> npt.NDArray[np.int32]:
    """Return number of DOFs per vertex.

    Args:
        n: Current number of DOFs per vertex.
        nmax: Mapping from configuration/archetype to maximum number of DOFs.

    Note:
        Only to be used for parageom example.

    """
    ndofs = {}
    for cfg, max_dofs in nmax.items():
        ndofs[cfg] = n if n <= max_dofs else max_dofs

    nl = ndofs.get('left')
    ni = ndofs.get('inner')
    nr = ndofs.get('right')

    dofs_per_vert = np.array([
        [nl, nl, nl, nl],
        [nl, ni, nl, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, nr, ni, nr],
        [nr, nr, nr, nr]], dtype=np.int32)
    return dofs_per_vert


if __name__ == "__main__":
    import pathlib
    import tempfile
    from multi.preprocessing import create_rectangle
    from multi.io import read_mesh
    from mpi4py import MPI

    gdim = 2
    with tempfile.NamedTemporaryFile(suffix=".msh") as tf:
        create_rectangle(0., 10., 0., 1., num_cells=(10, 1), recombine=True, out_file=tf.name)
        domain = read_mesh(pathlib.Path(tf.name), MPI.COMM_WORLD, kwargs={"gdim": gdim})[0]
    quadgrid = StructuredQuadGrid(domain)
    dofmap = GFEMDofMap(quadgrid)

    assert np.isclose(dofmap.num_cells, 10)
    assert np.isclose(dofmap.num_vertices, 22)

    # homogeneous dofmap
    dofs_per_vert = 4
    dofmap.distribute_dofs(dofs_per_vert)
    assert np.isclose(dofmap.num_dofs, dofs_per_vert * dofmap.num_vertices)
    assert np.allclose(dofmap.entity_dofs(0), np.arange(dofs_per_vert))
    assert np.allclose(dofmap.entity_dofs(21), np.arange(dofmap.num_dofs)[-dofs_per_vert:])

    # heterogeneous dofmap
    nl, ni, nr = 3, 5, 4
    dofs_per_vert = np.array([
        [nl, nl, nl, nl],
        [nl, ni, nl, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, ni, ni, ni],
        [ni, nr, ni, nr],
        [nr, nr, nr, nr]], dtype=np.int32)
    dofmap.distribute_dofs(dofs_per_vert)
    breakpoint()
    assert np.isclose(dofmap.num_dofs, 4 * nl + 4 * nr + (dofmap.num_vertices-8) * ni)
    assert np.isclose(len(dofmap.entity_dofs(0)), nl)
    assert np.isclose(len(dofmap.entity_dofs(5)), ni)


    dd = parageom_dof_distribution_factory(5, {'left': 3, 'inner': 5, 'right': 4})
    assert np.allclose(dofs_per_vert, dd)

    dofs_per_vert = 3
    dd = parageom_dof_distribution_factory(dofs_per_vert, {'left': 3, 'inner': 5, 'right': 4})
    dofmap.distribute_dofs(dd)
    assert np.isclose(dofmap.num_dofs, dofs_per_vert * dofmap.num_vertices)
    assert np.allclose(dofmap.entity_dofs(0), np.arange(dofs_per_vert))
    assert np.allclose(dofmap.entity_dofs(21), np.arange(dofmap.num_dofs)[-dofs_per_vert:])
