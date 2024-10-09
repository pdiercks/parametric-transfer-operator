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
        for method in example.methods:
            if np.isclose(cli.scale, 1.0):
                infile = example.projection_error(cli.nreal, method, cli.k, 1)
            else:
                infile = example.projection_error(cli.nreal, method, cli.k, cli.scale)
            data = np.load(infile)
            for key in ['l2_err_energy', 'relerr_energy']:
                e = data[key]
                if method == 'hrrf':
                    label = 'HRRF'
                    color = blue
                elif method == 'hapod':
                    label = 'RRF+POD'
                    color = red
                if key == 'l2_err_energy':
                    marker = 'x'
                elif key == 'relerr_energy':
                    marker = 'o'
                ax.semilogy(np.arange(e.size), e, marker=marker, markevery=5, color=color, label=label)
        ax.set_xlabel('Number of basis functions')
        ax.set_ylabel('Projection error')

        ax.set_ylim(1e-12, 5.0)
        yticks = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
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
