import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from .tasks import beam

    bamcd = read_bam_colors()

    markers = {
        "multivariate_normal": "x",
        "normal": "o",
    }  # use different markers per distribution
    colors = {
        "bottom": bamcd["red"][0],
        "left": bamcd["blue"][0],
        "right": bamcd["green"][0],
        "top": bamcd["yellow"][0],
    }

    with PlottingContext([__file__, args.output], [beam.plotting_style.as_posix()]) as fig:
        width = 3.149 # 1/2 of the printing box width
        height = 1.946
        fig.set_size_inches(width, height)
        ax = fig.subplots()

        config = args.configuration
        name = args.name
        distr = beam.distributions[0]

        infile = beam.proj_error(distr, config, name)
        npz_err = np.load(infile)
        marker = markers[distr]

        ymin = []

        for edge, color in colors.items():

            err = npz_err[edge]
            num_modes = np.arange(err.size)
            ax.semilogy(num_modes, err, color=color, marker=marker, label=edge)
            ymin.append(err[-1])
            # ax.semilogy(
            #         data["num_dofs"], data["err"],
            #         color=clr, marker=mark, label=label
            #         )
            # ax.fill_between(data["num_dofs"], data["err"]-std, data["err"]+std,
            #         alpha=0.2, color=clr
            #         )

        ax.set_ylim(min(ymin), 5.0)
        exponents = np.linspace(-12, 0, 7)
        base = np.array([10], dtype=float)
        ax.set_yticks(np.power(base, exponents))
        ax.set_xlabel("Number of basis functions.")
        # numerator = r"\norm{u_{\mathrm{fom}} - u_{\mathrm{rom}}}"
        # denominator = r"\norm{u_{\mathrm{fom}}}"
        # ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))
        ax.set_ylabel("Projection error.")  # TODO: add formula
        ax.legend(loc="best")


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configuration",
        type=str,
        help="The type of oversampling problem.",
        choices=("inner", "left", "right"),
    )
    parser.add_argument(
        "name",
        type=str,
        help="The name of the training strategy.",
        choices=("hapod", "heuristic"),
    )
    parser.add_argument("--output", type=str, help="Write plot to filepath.")
    args = parser.parse_args(sys.argv[1:])
    main(args)
