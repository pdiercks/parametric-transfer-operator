import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(cli):
    from parageom.tasks import example

    bamcd = read_bam_colors()
    blue = bamcd['blue'][0]
    red = bamcd['red'][0]

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()

        def add_relerr_plot(data, color, label):
            modes = np.arange(data['min_relerr_energy'].size)
            min = data['min_relerr_energy']
            max = data['max_relerr_energy']
            avg = data['avg_relerr_energy']
            ax.semilogy(modes, avg, color=color, linestyle='dashed', marker='.', markevery=5)
            ax.semilogy(modes, max, color=color, linestyle='solid', marker='o', markevery=5, label=label)
            ax.fill_between(modes, min, max, alpha=0.2, color=color)

        infile_hapod = example.projection_error(cli.nreal, 'hapod', cli.k, cli.scale)
        infile_hrrf = example.projection_error(cli.nreal, 'hrrf', cli.k, cli.scale)
        data_hapod = np.load(infile_hapod)
        data_hrrf = np.load(infile_hrrf)

        # l2-mean error in energy norm
        l2_hapod = data_hapod['l2_err_energy']
        l2_hrrf = data_hrrf['l2_err_energy']
        ax.semilogy(np.arange(l2_hapod.size), l2_hapod, marker='x', markevery=5, color=red, label='RRF+POD')
        ax.semilogy(np.arange(l2_hrrf.size), l2_hrrf, marker='x', markevery=5, color=blue, label='HRRF')

        # relative error in energy norm (min, avg, max)
        add_relerr_plot(data_hapod, red, 'RRF+POD')
        add_relerr_plot(data_hrrf, blue, 'HRRF')

        ax.set_xlabel('Number of basis functions')
        ax.set_ylabel('Projection error')

        ax.set_ylim(1e-8, 5.0)
        yticks = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        ax.set_yticks(yticks)

        ax.legend(loc='best')
        ax.grid(True, which='both')
        ax.set_title(rf'Rel. error (o) and $\ell^2$-mean error (x) for scale {cli.scale}')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument('k', type=int, help='The k-th realization.')
    parser.add_argument('scale', type=float, help='The scale of normal distribution.')
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
