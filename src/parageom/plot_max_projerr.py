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

    nmarks = 5
    if cli.k == 0:
        nmarks = 10
    elif cli.k == 5:
        nmarks = 20

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()

        def add_relerr_plot(data, color):
            modes = np.arange(data['mean_min_relerr_energy'].size)
            mean = data['mean_max_relerr_energy']
            std = data['std_max_relerr_energy']
            ax.semilogy(modes, mean, color=color, linestyle='solid', marker='s', markevery=nmarks)
            ax.fill_between(modes, mean - std, mean + std, alpha=0.2, color=color)

        infile_hapod = example.mean_projection_error('hapod', cli.k, cli.scale)
        infile_hrrf = example.mean_projection_error('hrrf', cli.k, cli.scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)

        # relative error in energy norm (min, avg, max)
        add_relerr_plot(data_hapod, red)  # , 'RRF+POD')
        add_relerr_plot(data_hrrf, blue)  # , 'HRRF')

        ax.set_xlabel(r'Local basis size $n$')
        ax.set_ylabel(r'Projection error $\mathcal{E}_{P}$')

        ax.set_ylim(1e-6, 5.0)
        yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(yticks)

        ax.legend(loc='best')
        ax.grid(True)

        red_line = Line2D([], [], color=red, marker='None', linestyle='-', label='RRFPOD')
        blue_line = Line2D([], [], color=blue, marker='None', linestyle='-', label='HRRF')
        max_marker = Line2D([], [], color='black', marker='s', linestyle='solid', label='max')

        fig.legend(
            handles=[max_marker, red_line, blue_line],
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
