import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors
from pymor.core.pickle import load


def main(args):
    from parageom.tasks import example

    if 'mdeim' in args.infile:
        with open(args.infile, 'rb') as fh:
            data = load(fh)
        svals = data['svals']
    else:
        svals = np.load(args.infile)
    nmodes = np.arange(1, svals.size + 1)
    sigma = svals / svals[0]

    bamcd = read_bam_colors()
    blue = bamcd['blue'][1]
    style = [example.plotting_styles['thesis-halfwidth'].as_posix()]
    with PlottingContext([__file__, args.outfile], style) as fig:
        ax = fig.subplots()
        ax.semilogy(nmodes, sigma, color=blue, linestyle='solid', marker='x')
        ax.set_xticks(nmodes)
        ax.set_xlabel(r'$n$')
        ax.set_ylabel(r'$\sigma_n$')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args)
