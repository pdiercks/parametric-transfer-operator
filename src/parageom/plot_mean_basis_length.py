import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from parageom.tasks import example

    bamcd = read_bam_colors()
    red = bamcd['red'][0]
    blue = bamcd['blue'][0]

    hapod = np.load(args.hapod)
    bs = {'hrrf': np.load(args.hrrf), 'hapod': hapod['num_modes']}
    snapshots = hapod['num_snapshots']
    configs = np.arange(1, 12, 1)
    barwidth = 0.35
    pos = {'hrrf': [x + barwidth / 2 for x in configs], 'hapod': [x - barwidth / 2 for x in configs]}

    style = [example.plotting_styles['thesis-halfwidth'].as_posix()]
    with PlottingContext([__file__, args.outfile], style) as fig:
        ax = fig.subplots()

        ax.bar(pos['hapod'], bs['hapod'], width=barwidth, color=red, label='RRFPOD')
        ax.bar(pos['hapod'], snapshots, width=barwidth, color=red, alpha=0.2)
        ax.bar(pos['hrrf'], bs['hrrf'], width=barwidth, color=blue, label='HRRF')

        ax.set_ylabel('Local basis size')
        ax.set_xlabel(r'Configuration $k$')
        ax.set_xticks(configs)
        ax.set_yscale('log')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('hapod', type=str)
    parser.add_argument('hrrf', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args)
