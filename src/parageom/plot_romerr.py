import numpy as np
from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext


def postproc(nreal: int, method: str, distr: str):
    from .tasks import example

    instream = example.locrom_error(nreal, method, distr)
    data = np.load(instream)
    max_dpv = np.load(example.local_basis_dofs_per_vert(nreal, method, distr))

    # now check values in data
    # and max dofs per vert

    breakpoint()


def main(cli):
    from .tasks import example

    distr = example.distributions[0]
    bamcd = read_bam_colors()
    blue = bamcd["blue"][0]
    red = bamcd["red"][0]

    marker = {
        "l2_err": "o",
        "max_relerr_h1_semi": "s",
        "max_relerr_max": "+",
    }
    # marker_ei = {
    #     "l2_err": "o",
    #     "max_relerr_h1_semi": "o",
    #     "max_relerr_max": "o",
    # }
    norm_label = {"max_relerr_h1_semi": "H1-semi", "max_relerr_max": "max"}

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for method in example.methods:
            infile = example.locrom_error(cli.nreal, method, distr)
            data = np.load(infile)
            num_dofs = data["ndofs"]
            for norm in cli.norm:
                err = data[norm]
                mark = marker[norm]
                if method == "heuristic":
                    label = f"HRRF, {norm_label[norm]}"
                    color = blue
                elif method == "hapod":
                    label = f"RRF+POD, {norm_label[norm]}"
                    color = red
                ax.semilogy(num_dofs, err, color=color, marker=mark, label=label)  # type: ignore

            if cli.ei:
                infile = example.locrom_error(cli.nreal, method, distr, ei=True)
                data = np.load(infile)
                num_dofs = data["ndofs"]
                for norm in cli.norm:
                    err = data[norm]
                    mark = marker[norm]
                    if method == "heuristic":
                        label = f"HRRF+EI, {norm_label[norm]}"
                        color = blue
                    elif method == "hapod":
                        label = f"RRF+POD+EI, {norm_label[norm]}"
                        color = red
                    ax.semilogy(num_dofs, err, color=color, linestyle="dotted", marker=mark, label=label)  # type: ignore
        ax.legend(loc="best")  # type: ignore
        ax.set_xlabel("Number of DOFs")  # type: ignore
        ax.set_ylabel("Relative error")  # type: ignore


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("--norm", nargs="+", type=str, required=True, help="Plot relative error in given norms")
    parser.add_argument("--ei", action="store_true", help="Additionally plot error of loc ROM with EI.")
    parser.add_argument("outfile", type=str, help="Write plot to path (pdf).")
    args = parser.parse_args(sys.argv[1:])
    main(args)
