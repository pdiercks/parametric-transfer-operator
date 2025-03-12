import numpy as np
from matplotlib.lines import Line2D
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(cli):
    """Plot mean max + standard deviation."""
    from parageom.tasks import example

    bamcd = read_bam_colors()
    blue = bamcd['blue'][0]
    red = bamcd['red'][0]

    # data
    # (a) config 6, mean_max_relerr_energy, std_max_relerr_energy
    # (b) config 1, mean_max_relerr_energy, std_max_relerr_energy
    # (c) config 1, mean_neumann_max_relerr_energy, std_neumann_max_relerr_energy

    scale = 0.1

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()

        def add_relerr_plot(config, data, color, marker, nmarks, neumann=False):
            if neumann:
                modes = np.arange(data['mean_neumann_max_relerr_energy'].size)
                mean = data['mean_neumann_max_relerr_energy']
                std = data['std_neumann_max_relerr_energy']
                ax.semilogy(modes, mean, color=color, linestyle='solid', marker=marker, markevery=nmarks)
                ax.fill_between(modes, mean - std, mean + std, alpha=0.2, color=color)
            else:
                modes = np.arange(data['mean_max_relerr_energy'].size)
                mean = data['mean_max_relerr_energy']
                std = data['std_max_relerr_energy']
                ax.semilogy(modes, mean, color=color, linestyle='solid', marker=marker, markevery=nmarks)
                ax.fill_between(modes, mean - std, mean + std, alpha=0.2, color=color)

        # (a) config 6, mean_max_relerr_energy, std_max_relerr_energy
        infile_hapod = example.mean_projection_error('hapod', 5, scale)
        infile_hrrf = example.mean_projection_error('hrrf', 5, scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)
        add_relerr_plot(5, data_hapod, red, 's', 20)
        add_relerr_plot(5, data_hrrf, blue, 's', 20)
        # (b) config 1, mean_max_relerr_energy, std_max_relerr_energy
        infile_hapod = example.mean_projection_error('hapod', 0, scale)
        infile_hrrf = example.mean_projection_error('hrrf', 0, scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)
        add_relerr_plot(0, data_hapod, red, 'o', 10)
        add_relerr_plot(0, data_hrrf, blue, 'o', 10)
        # (c) config 1, mean_neumann_max_relerr_energy, std_neumann_max_relerr_energy
        add_relerr_plot(0, data_hapod, red, '^', 10, neumann=True)
        add_relerr_plot(0, data_hrrf, blue, '^', 10, neumann=True)

        ax.set_xlabel(r'Local basis size $n$')
        ax.set_ylabel(r'Projection error $\mathcal{E}_{P}$')

        ax.set_ylim(1e-6, 5.0)
        yticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(yticks)
        xticks = list(range(0, 260, 25))
        ax.set_xticks(xticks)

        red_line = Line2D([], [], color=red, marker='None', linestyle='-', label='RRFPOD')
        blue_line = Line2D([], [], color=blue, marker='None', linestyle='-', label='HRRF')
        marker_5 = Line2D([], [], color='black', marker='s', linestyle='solid', label=r'$k=6$')
        marker_1 = Line2D([], [], color='black', marker='o', linestyle='solid', label=r'$k=1$, sp')
        marker_1_neumann = Line2D([], [], color='black', marker='^', linestyle='solid', label=r'$k=1$, N')

        fig.legend(
            handles=[marker_5, marker_1, marker_1_neumann, red_line, blue_line],
        )


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
