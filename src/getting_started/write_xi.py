import numpy as np


def main(args):
    from .tasks import beam
    from .definitions import BeamProblem

    distr = args.distr
    config = beam.cell_to_config(args.cell)

    # coarse and fine scale basis functions
    basis = np.load(beam.local_basis_npz(distr, config))
    out = {}
    out["phi"] = basis["phi"]

    p = BeamProblem(beam.coarse_grid.as_posix(), beam.fine_grid.as_posix())
    active_edges = p.active_edges[args.cell]
    for edge in active_edges:
        out[edge] = basis[edge]

    np.savez(beam.xi_npz(distr, args.cell), **out)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("distr", type=str, help="The distribution used for sampling.", choices=("normal", "multivariate_normal"))
    parser.add_argument("cell", type=int, help="The coarse grid cell index.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
