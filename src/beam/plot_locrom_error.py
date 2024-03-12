import sys
import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


if __name__ == "__main__":
    from .tasks import beam

    bamcd = read_bam_colors()
    color = {'hapod': bamcd["red"][0], "heuristic": bamcd["blue"][0]}

    with PlottingContext(sys.argv, ["paper_onecol"]) as fig:
        fig.set_size_inches(4.773, 2.95)
        ax = fig.subplots()
        ax.set_xlabel("Number of fine scale basis functions per edge.")
        ax.set_ylabel("locROM error relative to FOM.")

        for distribution in beam.distributions:
            for name in beam.training_strategies:
                infile = beam.loc_rom_error(distribution, name)
                data = np.loadtxt(infile.as_posix(), delimiter=",")
                modes = data[:, 0]
                error = data[:, 1]
                ax.semilogy(modes, error, color=color[name], marker="o", label=name)

        ax.legend(loc="best")



