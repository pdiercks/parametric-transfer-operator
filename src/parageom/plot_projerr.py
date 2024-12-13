import numpy as np
from matplotlib.lines import Line2D
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(cli):
    """Plot average values of the min, avg and max (over the validation set) projection error."""
    from parageom.tasks import example

    bamcd = read_bam_colors()
    blue = bamcd['blue'][0]
    red = bamcd['red'][0]

    # what is more important?
    # (a) show the standard deviation (realizations) for the max over the validation set?
    # (b) show the min, avg, max over the validation set, but leave out standard deviation for each?
    # --> For the projection error study we want to show that HRRF yields same quality
    # as HAPOD compression when it comes to the parameter variation.
    # The spread (difference between min and max) should indicate that all regions of
    # the parameter space are sufficiently covered. If this was not the case,
    # then the spread should be higher. (Assuming the validation set is sufficiently large).

    # Conclusion
    # Plot same as description in __docstring__.
    # The l2-mean error is not needed.

    nmarks = 5
    if cli.k == 0:
        nmarks = 5
    elif cli.k == 5:
        nmarks = 10

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()

        def add_relerr_plot(data, color):
            modes = np.arange(data['mean_min_relerr_energy'].size)
            min = data['mean_min_relerr_energy']
            max = data['mean_max_relerr_energy']
            avg = data['mean_avg_relerr_energy']
            ax.semilogy(modes, min, color=color, linestyle='None', marker='.', markevery=nmarks)
            ax.semilogy(modes, avg, color=color, linestyle='dashed', marker='o', markevery=nmarks)
            ax.semilogy(modes, max, color=color, linestyle='solid', marker='s', markevery=nmarks)
            ax.fill_between(modes, min, max, alpha=0.2, color=color)

        infile_hapod = example.mean_projection_error('hapod', cli.k, cli.scale)
        infile_hrrf = example.mean_projection_error('hrrf', cli.k, cli.scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)

        # relative error in energy norm (min, avg, max)
        add_relerr_plot(data_hapod, red)  # , 'RRF+POD')
        add_relerr_plot(data_hrrf, blue)  # , 'HRRF')

        ax.set_xlabel('Number of basis functions')
        ax.set_ylabel('Projection error')

        ax.set_ylim(1e-8, 5.0)
        yticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(yticks)

        ax.legend(loc='best')
        ax.grid(True, which='both')

        red_line = Line2D([], [], color=red, marker='None', linestyle='-', label='RRF+POD')
        blue_line = Line2D([], [], color=blue, marker='None', linestyle='-', label='HRRF')
        min_marker = Line2D([], [], color='black', marker='.', linestyle='None', label='min')
        avg_marker = Line2D([], [], color='black', marker='o', linestyle='dashed', label='avg')
        max_marker = Line2D([], [], color='black', marker='s', linestyle='solid', label='max')

        fig.legend(
            handles=[min_marker, avg_marker, max_marker, red_line, blue_line],
        )


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('k', type=int, help='The k-th transfer problem.')
    parser.add_argument('scale', type=float, help='The scale of normal distribution.')
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
