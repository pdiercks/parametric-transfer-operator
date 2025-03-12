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

    k = 0
    scale = 0.1
    nmarks = 10

    errtype = ['mean_{}_relerr_energy', 'mean_neumann_{}_relerr_energy']

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        axes = fig.subplots(1, 2, sharey=True)

        def add_relerr_plot(index, data, color, etype):
            modes = np.arange(data[etype.format('min')].size)
            min = data[etype.format('min')]
            max = data[etype.format('max')]
            avg = data[etype.format('avg')]
            ax = axes[index]
            ax.semilogy(modes, min, color=color, linestyle='None', marker='.', markevery=nmarks)
            ax.semilogy(modes, avg, color=color, linestyle='dashed', marker='o', markevery=nmarks)
            ax.semilogy(modes, max, color=color, linestyle='solid', marker='s', markevery=nmarks)
            ax.fill_between(modes, min, max, alpha=0.2, color=color)

        infile_hapod = example.mean_projection_error('hapod', k, scale)
        infile_hrrf = example.mean_projection_error('hrrf', k, scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)
        # breakpoint()

        for i, etype in enumerate(errtype):
            add_relerr_plot(i, data_hapod, red, etype)
            add_relerr_plot(i, data_hrrf, blue, etype)

        axes[0].set_xlabel(r'Local basis size $n$')
        axes[1].set_xlabel(r'Local basis size $n$')
        axes[0].set_ylabel(r'Projection error $\mathcal{E}_{P}$')

        ylim = (1e-10, 2.0)
        axes[0].set_ylim(*ylim)
        axes[1].set_ylim(*ylim)
        xticks = [0, 20, 40, 60, 80, 100]
        axes[0].set_xticks(xticks)
        axes[1].set_xticks(xticks)
        yticks = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
        axes[0].set_yticks(yticks)
        axes[1].set_yticks(yticks)

        # FIXME: minor tick labels are not shown on shared y-axis

        red_line = Line2D([], [], color=red, marker='None', linestyle='-', label='RRFPOD')
        blue_line = Line2D([], [], color=blue, marker='None', linestyle='-', label='HRRF')
        min_marker = Line2D([], [], color='black', marker='.', linestyle='None', label='min')
        avg_marker = Line2D([], [], color='black', marker='o', linestyle='dashed', label='avg')
        max_marker = Line2D([], [], color='black', marker='s', linestyle='solid', label='max')

        fig.legend(
            handles=[min_marker, avg_marker, max_marker, red_line, blue_line],
            loc='lower left',
            bbox_to_anchor=(0, 0),
            bbox_transform=axes[0].transAxes,
        )


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
