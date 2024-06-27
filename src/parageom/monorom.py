"""Build single domain ROM"""

from collections import defaultdict
from mpi4py import MPI  
import dolfinx as df
from dolfinx.io import gmshio

from pymor.algorithms.pod import pod
from pymor.reductors.basic import StationaryRBReductor
from pymor.bindings.fenicsx import FenicsxVectorSpace, FenicsxMatrixOperator, FenicsxVisualizer

from multi.boundary import plane_at, point_at
from multi.preprocessing import create_meshtags
from multi.domain import RectangularDomain
from multi.materials import LinearElasticMaterial
from multi.problems import LinearElasticityProblem
from multi.product import InnerProduct

import numpy as np


def main():
    """Build a monolithic ROM using standard FOM and POD."""
    from .tasks import example
    from .fom import discretize_fom
    from .auxiliary_problem import discretize_auxiliary_problem
    from .lhs import sample_lhs

    # ### FOM
    coarse_grid_path = example.coarse_grid("global").as_posix()
    parent_domain_path = example.parent_domain("global").as_posix()
    degree = example.geom_deg
    interface_tags = [
        i for i in range(15, 25)
    ]  # FIXME better define in Example data class
    aux = discretize_auxiliary_problem(
        parent_domain_path,
        degree,
        interface_tags,
        example.parameters["global"],
        coarse_grid=coarse_grid_path,
    )
    d = df.fem.Function(aux.problem.V, name="d_trafo")
    fom = discretize_fom(example, aux, d)

    # ### POD basis construction
    parameter_space = aux.parameters.space(example.mu_range)
    random_seed = example.training_set_seed
    num_snapshots = 100
    training_set = sample_lhs(parameter_space, name="R", samples=num_snapshots, criterion="center", random_state=random_seed)
    snapshots = fom.solution_space.empty(reserve=num_snapshots)
    breakpoint()

    for mu in training_set:
        snapshots.append(fom.solve(mu))
    basis, svals = pod(snapshots, product=fom.h1_0_semi_product, l2_err=0.001)
    reductor = StationaryRBReductor(
        fom, product=fom.h1_0_semi_product, check_orthonormality=False
    )
    reductor.extend_basis(basis, method="trivial")

    fom.visualizer.visualize(
        basis, filename="./work/parageom/monolithic_pod_basis.xdmf"
    )
    rom = reductor.reduce()

    errors = defaultdict(list)
    ntest = 10
    test_set = parameter_space.sample_randomly(ntest)
    for mu in test_set:
        fom_data = fom.compute(solution=True, output=False, mu=mu)
        rom_data = rom.compute(solution=True, output=False, mu=mu)
        for key in ("solution",):
            if key == "solution":
                ERR = fom_data.get(key) - reductor.reconstruct(rom_data.get(key))
                err = ERR.norm(fom.h1_0_semi_product)
            else:
                ERR = fom_data.get(key) - rom_data.get(key)
                err = ERR[0, 0]

            errors[key].append(err)

    for key in ("solution",):
        print(
            f"Max {key} error = {max(errors[key])} over test set of size {len(test_set)}"
        )


if __name__ == "__main__":
    # pull_back_rom()
    # physical_mesh_rom()
    main()
