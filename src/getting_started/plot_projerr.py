import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main():
    from .tasks import beam
    bamcd = read_bam_colors()

    markers = {"multivariate_normal": "x", "normal": "o"}  # use different markers per distribution
    colors = {"inner": bamcd["red"][0], "left": bamcd["blue"][0], "right": bamcd["green"][0]}  # use different colors per configuration

    for config in beam.configurations:
        outfile = beam.fig_proj_error(config)

        with PlottingContext([__file__, outfile.as_posix()], "fast") as fig:
            # figure.figsize: 4.773, 2.950
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

                label += config
                infile = beam.proj_error(distr, config)
                errors = np.load(infile)
                num_modes = np.arange(errors.size)

                color = colors[config]
                marker = markers[distr]
                ax.semilogy(num_modes, errors, color=color, marker=marker, label=label)
                # ax.semilogy(
                #         data["num_dofs"], data["err"],
                #         color=clr, marker=mark, label=label
                #         )
                # ax.fill_between(data["num_dofs"], data["err"]-std, data["err"]+std,
                #         alpha=0.2, color=clr
                #         )

            ax.set_xlabel("Number of basis functions.")
            # numerator = r"\norm{u_{\mathrm{fom}} - u_{\mathrm{rom}}}"
            # denominator = r"\norm{u_{\mathrm{fom}}}"
            # ax.set_ylabel(r"$\nicefrac{{{}}}{{{}}}$".format(numerator, denominator))
            ax.set_ylabel("Projection error.")  # TODO: add formula
            ax.legend(loc="best", bbox_to_anchor=(0.65, 1.0))



if __name__ == "__main__":
    main()
