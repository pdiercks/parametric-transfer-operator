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
            for key in ['l2err_h1_semi']:
                e = data[key]
                if method == 'heuristic':
                    label = 'HRRF'
                    color = blue
                elif method == 'hapod':
                    label = 'RRF+POD'
                    color = red
                    eps = np.ones_like(e) * example.epsilon_star_projerr**2
                    ax.semilogy(  # type: ignore
                        np.arange(e.size), eps, 'k-', label=r'$(\varepsilon^{\ast})^2$'
                    )
                ax.semilogy(np.arange(e.size), e, color=color, label=label)  # type: ignore
        ax.legend(loc='best')  # type: ignore
        ax.set_xlabel('Number of basis functions')  # type: ignore
        ax.set_ylabel(r'$\ell^2$-mean projection error')  # type: ignore


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument('k', type=int, help='The k-th realization.')
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
