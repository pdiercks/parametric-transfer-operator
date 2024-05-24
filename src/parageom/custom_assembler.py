# Copyright (C) 2020 Igor A. Baratta
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy as np
from scipy.sparse import coo_array
import collections
# import dolfinx.cpp.fem

import dolfinx as df
import numba
import cffi

# import numba.core.typing.cffi_utils as cffi_support

# TODO
# understand how numba, cffi, and petsc play together for custom assembly
# see imports/definitions in test_custom_assembler.py

# CFFI - register complex types
ffi = cffi.FFI()
# cffi_support.register_type(ffi.typeof(
#     "double _Complex"), numba.types.complex128)
# cffi_support.register_type(ffi.typeof("float _Complex"), numba.types.complex64)

DofMapWrapper = collections.namedtuple("DofMapWrapper", "dof_array num_cell_dofs size")
MeshWrapper = collections.namedtuple("MeshWrapper", "cell topology geometry")
TopologyWrapper = collections.namedtuple("TopologyWrapper", "dim num_cells num_facets")
GeometryWrapper = collections.namedtuple("GeometryWrapper", "dim x x_dofs pos")
CellWrapper = collections.namedtuple("CellWrapper", "num_vertices")


def dofmap_wrapper(dofmap: df.cpp.fem.DofMap):
    num_cell_dofs = dofmap.dof_layout.num_dofs
    dof_array = dofmap.map()
    size = np.max(dof_array) + 1
    return DofMapWrapper(dof_array, num_cell_dofs, size)


def mesh_wrapper(mesh: df.mesh.Mesh):
    topology = topology_wrapper(mesh.topology)
    geometry = geometry_wrapper(mesh)
    cell = CellWrapper(num_vertices=mesh.ufl_cell().num_vertices())
    return MeshWrapper(cell=cell, topology=topology, geometry=geometry)


def topology_wrapper(topology: df.cpp.mesh.Topology):
    dim = topology.dim
    for dd in range(dim):
        topology.create_connectivity(dim, dd)
    num_cells = topology.index_map(dim).size_local + topology.index_map(dim).num_ghosts
    num_facets = (
        topology.index_map(dim - 1).size_local + topology.index_map(dim - 1).num_ghosts
    )
    # FIXME: df.cpp.mesh.compute_boundary_facets does not exist
    # boundary_facets = np.where(np.array(df.cpp.mesh.compute_boundary_facets(topology)) == 1)[0]
    return TopologyWrapper(dim, num_cells, num_facets)


def geometry_wrapper(mesh: df.cpp.mesh.Mesh_float64):
    geometry = mesh.geometry
    dim = geometry.dim
    # crop points according to dimension
    x = geometry.x[:, :dim]
    x_dofs = geometry.dofmap
    tdim = mesh.topology.dim
    pos = mesh.topology.connectivity(tdim, 0).offsets
    return GeometryWrapper(dim, x, x_dofs, pos)


def assemble_matrix(
    domain, form, active_entities=None, form_compiler_options=None, jit_options=None
):
    if active_entities is None:
        active_entities = {}

    if form_compiler_options is None:
        form_compiler_options = {"scalar_type": df.default_scalar_type}

    # if jit_options is None:
    #     jit_options = {}

    dtype = form_compiler_options.get("scalar_type")

    # FIXME
    # why do need to compile twice? (fem.form vs ffcx_jit?)

    compiled_form = df.fem.form(
        form, form_compiler_options=form_compiler_options, jit_options=jit_options
    )
    # mesh = compiled_form.mesh

    # get ufcx_form from compiled_form
    ufcx_form = compiled_form.ufcx_form

    # ffcx_jit is a decorated function and requires MPI communicator
    # ufcx_form, _, _ = df.jit.ffcx_jit(
    #         mesh.comm, form, form_compiler_options=form_compiler_options, jit_options=jit_options
    #         )

    # compiled_form.mesh is not an instance of dolfinx.mesh.Mesh, but rather dolfinx.cpp.mesh.Mesh
    # mesh = mesh_wrapper(compiled_form.mesh)
    # cpp_mesh = compiled_form.mesh
    # instead of domain, might reduce to passing the ufl_cell only
    # but cpp_mesh does not have info on ufl cell is the problem
    mesh = mesh_wrapper(domain)

    # same goes for dofmap, dolfinx.cpp.fem.DofMap
    breakpoint()
    dofmap = dofmap_wrapper(compiled_form.function_spaces[0].dofmap)

    # TODO: is shape of data correct?
    # data.size should equal number of non-zero entries in the global matrix
    # check also sparsity pattern
    data = np.zeros(dofmap.num_cell_dofs * dofmap.dof_array.size, dtype=dtype)

    # pack_coefficients expects df.cpp.fem.Form_{dtype}
    # compiled_form is df.fem.Form

    # FIXME
    # pack_coefficients & pack_constants need to be tested for forms with coeffs & constants
    # those functions return dict with key (dolfinx.cpp.fem.IntegralType.cell, -1) or similar
    # and value of type np.ndarray
    # the np.ndarray needs to be passed to kernel in `assemble_cells`

    # Why not use df.fem.pack_constants etc.?
    coefficients = df.cpp.fem.pack_coefficients(compiled_form._cpp_object)
    constants = df.cpp.fem.pack_constants(compiled_form._cpp_object)
    # FIXME
    # when are permutations required?
    # should get this from basix?
    perm = np.array([0], dtype=np.uint8)

    # FIXME
    # allow form to have more than one integral
    # proper way of getting the integral type (required to get coeffs & constants)
    # check dolfinx.fem.form

    # FIXME
    coeffs = coefficients[(df.cpp.fem.IntegralType.cell, -1)]

    # dolfinx.fem.form
    # use ufl form to get subdomain data
    # get cpp object for each coeff and constant in the form

    # Now, assemble_vector etc. expect return value of dolfinx.fem.form
    # see there how cpp objects are passed to kernel

    # See, cpp version of assemble_vector etc.
    # Simply, modify those to loop over subset of cells only?

    # FIXME
    kernel = getattr(
        ufcx_form.form_integrals[0], f"tabulate_tensor_{np.dtype(dtype).name}"
    )
    # (Pdb) p kernel
    # 'void(*)(double *, double *, double *, double *, int *, uint8_t *)'

    active_cells = active_entities.get("cells", np.arange(mesh.topology.num_cells))
    assemble_cells(
        data, kernel, dofmap, mesh, coeffs, constants, perm, active_cells
    )

    # if ufcx_form.num_cell_integrals:
    #     active_cells = active_entities.get(
    #         "cells", numpy.arange(mesh.topology.num_cells))
    #     # create_cell_integral() does not exist???
    #     cell_integral = ufcx_form.create_cell_integral(-1)
    #     kernel = cell_integral.tabulate_tensor
    #     assemble_cells(
    #         data, kernel, dofmap, mesh, coefficients, constants, perm, active_cells,
    #     )

    # FIXME
    # allow form to contain exterior facet integrals

    # if ufcx_form.num_exterior_facet_integrals:
    #     active_facets = active_entities.get(
    #         "facets", mesh.topology.boundary_facets)
    #     facet_data = facet_info(_a.mesh, active_facets)
    #     facet_integral = ufc_form.create_exterior_facet_integral(-1)
    #     kernel = facet_integral.tabulate_tensor
    #     assemble_facets(
    #         data, kernel, dofmap, mesh, coefficients, constants, perm, facet_data,
    #     )

    # FIXME
    # shape=(dofmap.size, dofmap.size) is always the full matrix
    # return smaller matrix if only some cells are active?
    # or does not matter because sparse format does not store zeroes?

    local_mat = coo_array(
        (data, sparsity_pattern(dofmap)), shape=(dofmap.size, dofmap.size)
    ).tocsr()

    return local_mat


# @numba.njit(fastmath=True)
def assemble_cells(
    data,
    kernel,
    dofmap: DofMapWrapper,
    mesh: MeshWrapper,
    coeffs,
    constants,
    perm,
    active_cells,
):
    (_, x, x_dofs, pos) = mesh.geometry
    entity_local_index = np.array([0], dtype=np.int32)
    local_mat = np.zeros((dofmap.num_cell_dofs, dofmap.num_cell_dofs), dtype=data.dtype)
    for idx in active_cells:
        coordinate_dofs = x[x_dofs[pos[idx] : pos[idx + 1]], :]
        local_mat.fill(0.0)
        kernel(
            ffi.from_buffer(local_mat),
            ffi.from_buffer(coeffs[idx, :]),
            ffi.from_buffer(constants),
            ffi.from_buffer(coordinate_dofs),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )
        data[
            idx * local_mat.size : idx * local_mat.size + local_mat.size
        ] += local_mat.ravel()


# @numba.njit(fastmath=True)
# def assemble_facets(data, kernel, dofmap: DofMapWrapper, mesh: MeshWrapper, coeffs, constants, perm, facet_data):
#     entity_local_index = numpy.array([0], dtype=numpy.int32)
#     Ae = numpy.zeros(
#         (dofmap.num_cell_dofs, dofmap.num_cell_dofs), dtype=data.dtype)
#     gdim, x, x_dofs, pos = mesh.geometry
#     num_active_facets = facet_data.shape[0]
#     for i in range(num_active_facets):
#         local_facet, cell_idx = facet_data[i]
#         entity_local_index[0] = local_facet
#         coordinate_dofs = x[x_dofs[pos[cell_idx]: pos[cell_idx + 1]], :]
#         Ae.fill(0.0)
#         kernel(
#             ffi.from_buffer(Ae),
#             ffi.from_buffer(coeffs[cell_idx, :]),
#             ffi.from_buffer(constants),
#             ffi.from_buffer(coordinate_dofs),
#             ffi.from_buffer(entity_local_index),
#             ffi.from_buffer(perm),
#             0,
#         )
#         data[cell_idx * Ae.size: cell_idx * Ae.size + Ae.size] += Ae.ravel()


def sparsity_pattern(dofmap: DofMapWrapper):
    """
    Returns local COO sparsity pattern
    """
    num_cells = dofmap.dof_array.size // dofmap.num_cell_dofs
    rows = np.repeat(dofmap.dof_array, dofmap.num_cell_dofs)
    cols = np.tile(
        np.reshape(dofmap.dof_array, (num_cells, dofmap.num_cell_dofs)),
        dofmap.num_cell_dofs,
    )
    return rows, cols.ravel()


# def facet_info(mesh, active_facets):
#     # FIXME: Refactor this function using the wrapper
#     # get facet-cell and cell-facet connections
#     tdim = mesh.topology.dim
#     c2f = mesh.topology.connectivity(tdim, tdim - 1).array
#     c2f_offsets = mesh.topology.connectivity(tdim, tdim - 1).offsets
#     f2c = mesh.topology.connectivity(tdim - 1, tdim).array
#     f2c_offsets = mesh.topology.connectivity(tdim - 1, tdim).offsets
#     facet_data = numpy.zeros((active_facets.size, 2), dtype=numpy.int32)
#
#     @numba.njit(fastmath=True)
#     def facet2cell(data):
#         for j, facet in enumerate(active_facets):
#             cells = f2c[f2c_offsets[facet]: f2c_offsets[facet + 1]]
#             local_facets = c2f[c2f_offsets[cells[0]]: c2f_offsets[cells[0] + 1]]
#             local_facet = numpy.where(facet == local_facets)[0][0]
#             data[j, 0] = local_facet
#             data[j, 1] = cells[0]
#         return data
#
#     return facet2cell(facet_data)
