from collections import defaultdict

import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(cli):
    from parageom.tasks import example

    bamcd = read_bam_colors()
    blue = bamcd['blue'][0]
    red = bamcd['red'][0]

    nreal = cli.nreal
    number_of_modes = example.validate_rom['num_modes']

    erru = defaultdict(list)
    errs = defaultdict(list)

    # define data to gather via keys
    keys = ['relerr']

    # append 1.0 for num_modes=0
    for key in keys:
        erru['min_' + key].append(1.0)
        erru['max_' + key].append(1.0)
        erru['avg_' + key].append(1.0)
        errs['min_' + key].append(1.0)
        errs['max_' + key].append(1.0)
        errs['avg_' + key].append(1.0)

    for num_modes in number_of_modes:
        data_u = np.load(example.rom_error_u(nreal, num_modes, method=cli.method, ei=cli.ei))
        # data_s = np.load(example.rom_error_s(nreal, num_modes, method=cli.method, ei=cli.ei))

        for key in keys:
            erru['min_' + key].append(np.min(data_u[key]))
            erru['max_' + key].append(np.max(data_u[key]))
            erru['avg_' + key].append(np.average(data_u[key]))

            # errs["min_" + key].append(np.min(data_s[key]))
            # errs["max_" + key].append(np.max(data_s[key]))
            # errs["avg_" + key].append(np.average(data_s[key]))

    number_of_modes = [
        0,
    ] + number_of_modes

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        ax.semilogy(number_of_modes, erru['avg_relerr'], color=red, linestyle='dashed', marker='.')  # type: ignore
        # ax.semilogy(number_of_modes, errs["avg_relerr"], color=blue, linestyle="dashed", marker=".")  # type: ignore
        ax.fill_between(number_of_modes, erru['min_relerr'], erru['max_relerr'], alpha=0.2, color=red)  # type: ignore
        # ax.fill_between(number_of_modes, errs["min_relerr"], errs["max_relerr"], alpha=0.2, color=blue)  # type: ignore
        ax.semilogy(number_of_modes, erru['max_relerr'], color=red, linestyle='solid', marker='o', label=r'$e_u$')  # type: ignore
        # ax.semilogy( # type: ignore
        #     number_of_modes, errs["max_relerr"], color=blue, linestyle="solid", marker="o", label=r"$e_{\sigma}$"
        # )
        ax.legend(loc='best')  # type: ignore
        ax.set_xlabel('Local basis size')  # type: ignore
        ax.set_ylabel('Relative error')  # type: ignore


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('nreal', type=int, help='The n-th realization.')
    parser.add_argument(
        'method',
        type=str,
        help='The method used for basis construction.',
        choices=('hapod', 'heuristic'),
        default='hapod',
    )
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    parser.add_argument('--ei', action='store_true', help='Plot data of validation of ROM with EI.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
