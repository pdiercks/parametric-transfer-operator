import numpy as np
from multi.plotting_context import PlottingContext


def main(cli):
    from collections import defaultdict

    from pymor.core.pickle import load

    from parageom.tasks import example

    # Extract HAPOD data (num_modes / num_snapshots)
    # and take the average over realizations for each transfer problem
    keys = ['num_snapshots', 'num_modes']
    average = defaultdict(list)
    for j in range(11):
        local_values = defaultdict(list)
        for nreal in range(example.num_real):
            with example.hapod_summary(nreal, j).open('rb') as fh:
                data = load(fh)
                for k in keys:
                    local_values[k].append(data[k])
        for k in keys:
            assert local_values[k].size == example.num_real
            average[k] = np.average(local_values[k])

    def myfrac(data, index):
        num_modes = data['num_modes'][index]
        num_snaps = data['num_snapshots'][index]
        return str(num_modes), str(num_snaps)

    # FIXME
    # averaging the singular values is rubbish of course
    # Thus, we simply plot the singular values for a single realization with the aim to discuss rapid decay, i.e., that HAPOD nicely compresses the data.
    # Then, we provide average values for HAPOD (modes / snapshots) in comparison with HRRF (number of iterations) in a table to discuss the computational costs for the basis construction.

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for k in range(11):
            numerator, denominator = myfrac(average, k)
            nicefrac = r'\nicefrac{{{}}}{{{}}}'.format(numerator, denominator)
            sigma = np.load(example.hapod_singular_values(cli.nreal, k))
            modes = np.arange(sigma.size)
            label = rf'$k={k+1}$, ${nicefrac}$'
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
