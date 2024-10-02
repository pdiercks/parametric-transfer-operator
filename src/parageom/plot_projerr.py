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
            infile = example.projerr(cli.nreal, method, cli.k)
            data = np.load(infile)
            for key in ['l2_err_energy', 'relerr_energy']:
                e = data[key]
                if method == 'heuristic':
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
        ax.legend(loc='best')  # type: ignore
        ax.set_xlabel('Number of basis functions')
        ax.set_ylabel(r'$\ell^2$-mean projection error')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument('k', type=int, help='The k-th realization.')
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
