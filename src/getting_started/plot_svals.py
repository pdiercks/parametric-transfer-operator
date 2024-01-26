import sys
import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from .tasks import beam

    bamcd = read_bam_colors()
    markers = {"multivariate_normal": "x", "normal": "o"}
    colors = {
        "inner": bamcd["red"][0],
        "left": bamcd["blue"][0],
        "right": bamcd["green"][0],
    }
    config = args.pop(2)

    with PlottingContext(args, ["paper_onecol"]) as fig:
        width = 4.773
        height = 2.95
        factor = 1.0
        fig.set_size_inches(factor * width, factor * height)
        ax = fig.subplots()

        for distr in beam.distributions:
            label = ""
            if distr == "multivariate_normal":
                label = "correlated, "
            if distr == "normal":
                label = "uncorrelated, "

            svals = np.load(beam.loc_singular_values(distr, config))

            color = colors[config]
            marker = markers[distr]
            label += config
            ax.semilogy(
                np.arange(svals.size),
                svals / svals[0],
                color=color,
                marker=marker,
                label=label,
            )

        ax.set_xlabel("Number of basis functions")
        ax.set_ylabel("Singular values")  # TODO: add formula
        ax.legend(loc="best")


if __name__ == "__main__":
    main(sys.argv)
