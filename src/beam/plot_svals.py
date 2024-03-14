import sys
import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from .tasks import beam

    bamcd = read_bam_colors()
    markers = {"multivariate_normal": "x", "normal": "o"}
    colors = {
        "bottom": bamcd["red"][0],
        "left": bamcd["blue"][0],
        "right": bamcd["green"][0],
        "top": bamcd["yellow"][0],
    }
    config = args.pop(2)

    with PlottingContext(args, [beam.plotting_style.as_posix()]) as fig:
        width = 3.149 # 1/2 of the printing box width
        height = 1.946
        fig.set_size_inches(width, height)
        ax = fig.subplots()

        for distr in beam.distributions:
            svals = np.load(beam.loc_singular_values_npz(distr, config))

            for edge, color in colors.items():

                sigma = svals[edge]
                marker = markers[distr]
                ax.semilogy(
                    np.arange(sigma.size),
                    sigma / sigma[0],
                    color=color,
                    marker=marker,
                    label=edge,
                )

        ax.set_xlabel("Number of basis functions")
        ax.set_ylabel("Singular values")  # TODO: add formula
        ax.legend(loc="best")


if __name__ == "__main__":
    main(sys.argv)
