import numpy as np

from mpi4py import MPI
from dolfinx import fem, default_scalar_type
from dolfinx.io import gmshio, XDMFFile
import ufl


def compute_volume_physical(mshfile, trafod):
    domain = gmshio.read_from_msh(
        mshfile, MPI.COMM_WORLD, gdim=2
    )[0]

    x_subdomain = domain.geometry.x
    disp = np.pad(
        trafod.x.array.reshape(x_subdomain.shape[0], -1),  # type: ignore
        pad_width=[(0, 0), (0, 1)],
    )
    x_subdomain += disp

    dx = ufl.Measure('dx', domain=domain)
    vol = 1. * dx
    volume = fem.assemble_scalar(fem.form(vol))
    return volume


def compute_parent_volume(domain, trafod):
    Id = ufl.Identity(2)
    F = Id + ufl.grad(trafod)  # type: ignore
    detF = ufl.det(F)
    dx = ufl.Measure('dx', domain=domain)
    vol = detF * dx
    volume = fem.assemble_scalar(fem.form(vol))
    return volume


def main():
    from .tasks import example
    from .auxiliary_problem import discretize_auxiliary_problem

    # Generate physical subdomain
    parent_subdomain_msh = example.parent_unit_cell.as_posix()
    degree = example.geom_deg

    aux = discretize_auxiliary_problem(
        parent_subdomain_msh, degree, example.parameters["subdomain"]
    )
    radius = 0.29544
    mu = aux.parameters.parse([radius])
    d = fem.Function(aux.problem.V)
    aux.solve(d, mu)  # type: ignore

    vol_ana = 1. - radius ** 2 * np.pi
    vol_phys = compute_volume_physical(parent_subdomain_msh, d)
    vol_parent = compute_parent_volume(d.function_space.mesh, d)
    print(f"{vol_ana=}")
    print(f"{vol_phys=}")
    print(f"{vol_parent=}")
    rel_err = (vol_phys - vol_parent) / vol_phys
    print(f"{rel_err=}")


if __name__ == "__main__":
    main()
