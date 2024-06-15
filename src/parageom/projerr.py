"""compute projection error to assess quality of the basis"""

import numpy as np

from multi.projection import project_array, orthogonal_part

from pymor.core.logger import getLogger
from pymor.core.defaults import set_defaults
from pymor.tools.random import new_rng


def main(args):
    from .tasks import example
    from .lhs import sample_lhs
    from .locmor import discretize_transfer_problem

    logfilename = example.log_projerr(
        args.nreal, args.method, args.distr, args.config
    ).as_posix()
    set_defaults({"pymor.core.logger.getLogger.filename": logfilename})
    logger = getLogger("projerr", level="INFO")

    transfer, f_ext = discretize_transfer_problem(example, args.config)

    # ### Read basis and wrap as pymor object
    basis_path = None
    if args.method == "hapod":
        basis_path = example.hapod_modes_npy(args.nreal, args.distr, args.config)
    elif args.method == "heuristic":
        basis_path = example.heuristic_modes_npy(args.nreal, args.distr, args.config)
    else:
        raise NotImplementedError
    local_basis = np.load(basis_path)
    basis = transfer.range.from_numpy(local_basis)

    full_basis = transfer.kernel
    full_basis.append(basis)

    orthonormal = np.allclose(full_basis.gramian(range_product), np.eye(len(full_basis)), atol=1e-5)
    if not orthonormal:
        raise ValueError("Basis is not orthonormal wrt range product.")

    # Definition of validation set
    # make sure that this is always the same set of parameters
    # and also same set of boundary data
    # but different from μ and g used in the training
    parameter_space = transfer.operator.parameters.space(example.mu_range)
    parameter_name = list(example.parameters[args.config].keys())[0]
    test_set = sample_lhs(
        parameter_space,
        name=parameter_name,
        samples=30,
        criterion="center",
        random_state=example.projerr_seed
    )

    test_data = transfer.range.empty(reserve=len(test_set))

    logger.info(f"Computing test set of size {len(test_set)}...")
    with new_rng(example.projerr_seed):
        for mu in test_set:
            transfer.assemble_operator(mu)
            g = transfer.generate_random_boundary_data(1, "normal", {"scale": 0.1})
            test_data.append(transfer.solve(g))
            neumann = transfer.op.apply_inverse(f_ext)
            neumann_in = transfer.range.from_numpy(neumann.dofs(transfer._restriction))
            test_data.append(orthogonal_part(neumann_in, transfer.kernel, product=transfer.range_product, orthonormal=True))

    aerrs = []
    rerrs = []
    u_norm = test_data.norm(transfer.range_product) # norm of each test vector

    logger.info("Computing relative projection error ...")
    for N in range(len(full_basis) + 1):
        U_proj = project_array(test_data, full_basis[:N], product=transfer.range_product, orthonormal=orthonormal)
        err = (test_data - U_proj).norm(transfer.range_product) # absolute projection error
        if np.all(err == 0.):
            # ensure to return 0 here even when the norm of U is zero
            rel_err = err
        else:
            rel_err = err / u_norm
        aerrs.append(np.max(err))
        rerrs.append(np.max(rel_err))

    rerr = np.array(rerrs)
    aerr = np.array(aerrs)
    if args.output is not None:
        np.save(args.output, np.vstack((rerr, aerr)).T)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=str, help="The n-th realization.")
    parser.add_argument("method", type=str, help="Method used for basis construction.")
    parser.add_argument("distr", type=str, help="Distribution used for random sampling.")
    parser.add_argument("config", type=str, help="Configuration / Archetype.")
    parser.add_argument("--output", type=str, help="Write absolute and relative projection error to file.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
