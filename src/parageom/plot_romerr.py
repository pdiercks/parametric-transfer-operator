import numpy as np
from collections import defaultdict
from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext


def main(cli):
    from .tasks import example

    bamcd = read_bam_colors()
    blue = bamcd["blue"][0]
    red = bamcd["red"][0]

    nreal = cli.nreal
    number_of_modes = [20, 40, 60, 80, 100, 120, 140, 160]

    erru = defaultdict(list)
    errs = defaultdict(list)

    # define data to gather via keys
    keys = ["relerr"]

    # append 1.0 for num_modes=0
    for key in keys:
        erru["min_"+key].append(1.0)
        erru["max_"+key].append(1.0)
        erru["avg_"+key].append(1.0)
        errs["min_"+key].append(1.0)
        errs["max_"+key].append(1.0)
        errs["avg_"+key].append(1.0)

    for num_modes in number_of_modes:
        data_u = np.load(example.rom_error_u(nreal, num_modes, ei=cli.ei))
        data_s = np.load(example.rom_error_s(nreal, num_modes, ei=cli.ei))

        for key in keys:
            erru["min_"+key].append(np.min(data_u[key]))
            erru["max_"+key].append(np.max(data_u[key]))
            erru["avg_"+key].append(np.average(data_u[key]))

            errs["min_"+key].append(np.min(data_s[key]))
            errs["max_"+key].append(np.max(data_s[key]))
            errs["avg_"+key].append(np.average(data_s[key]))

    number_of_modes = [0,] + number_of_modes

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        ax.semilogy(number_of_modes, erru["avg_relerr"], color=red, linestyle="dashed", marker=".")
        ax.semilogy(number_of_modes, errs["avg_relerr"], color=blue, linestyle="dashed", marker=".")
        ax.fill_between(number_of_modes, erru["min_relerr"], erru["max_relerr"], alpha=0.2, color=red)
        ax.fill_between(number_of_modes, errs["min_relerr"], errs["max_relerr"], alpha=0.2, color=blue)
        ax.semilogy(number_of_modes, erru["max_relerr"], color=red, linestyle="solid", marker="o", label=r"$e_u$")  # type: ignore
        ax.semilogy(number_of_modes, errs["max_relerr"], color=blue, linestyle="solid", marker="o", label=r"$e_{\sigma}$")
        ax.legend(loc="best")  # type: ignore
        ax.set_xlabel("Local basis size")  # type: ignore
        ax.set_ylabel("Relative error")  # type: ignore


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("--ei", action="store_true", help="Plot data of validation of ROM with EI.")
    parser.add_argument("outfile", type=str, help="Write plot to path (pdf).")
    args = parser.parse_args(sys.argv[1:])
    main(args)
