import pathlib
import numpy as np


def main(args):
    from parageom.tasks import example

    hapod = np.load(args.hapod)
    bs = {'hrrf': np.load(args.hrrf), 'hapod': hapod['num_modes']}
    snapshots = hapod['num_snapshots']


    style = [example.plotting_styles['thesis']]
    with PlottingContext([__file__, args.outfile], style) as fig:


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("hapod", type=str)
    parser.add_argument("hrrf", type=str)
    parser.add_argument("outfile", type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args)
