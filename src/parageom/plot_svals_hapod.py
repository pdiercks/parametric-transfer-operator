import numpy as np
from multi.plotting_context import PlottingContext


def main(cli):
    from parageom.tasks import example

    # def myfrac(data, index):
    #     num_modes = data['num_modes'][index]
    #     num_snaps = data['num_snapshots'][index]
    #     return str(num_modes), str(num_snaps)

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for k in range(11):
            # numerator, denominator = myfrac(average, k)
            # nicefrac = r'\nicefrac{{{}}}{{{}}}'.format(numerator, denominator)
            sigma = np.load(example.hapod_singular_values(cli.nreal, k))
            modes = np.arange(sigma.size)
            # label = rf'$k={k+1}$, ${nicefrac}$'
            label = rf'$k={k+1}$'
            ax.semilogy(modes, sigma / sigma[0], linestyle='dashed', label=label)
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))  # type: ignore
        ax.set_xlabel(r'$n$')  # type: ignore
        ax.set_ylabel('Singular value')  # type: ignore


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser('Plot Singular values of each Oversampling Problem.')
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    args = parser.parse_args(sys.argv[1:])
    main(args)
