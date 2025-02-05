import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from parageom.tasks import example

    bamcd = read_bam_colors()
    red = bamcd['red'][0]
    blue = bamcd['blue'][0]

    color = {'hrrf': blue, 'hapod': red}
    label = {'hrrf': 'HRRF', 'hapod': 'RRFPOD'}

    data = {'hrrf': np.load(args.hrrf), 'hapod': np.load(args.hapod)}
    dofs = {}
    for key, item in data.items():
        dofs[key] = item['dofs']

    num_modes = np.array(example.rom_validation.num_modes) / 10
    barwidth = 0.35
    pos = {'hrrf': [x + barwidth / 2 for x in num_modes], 'hapod': [x - barwidth / 2 for x in num_modes]}

    style = [example.plotting_styles['thesis-halfwidth'].as_posix()]
    with PlottingContext([__file__, args.outfile], style) as fig:
        ax = fig.subplots()

        for method in example.methods:
            ax.bar(pos[method], dofs[method], width=barwidth, color=color[method], label=label[method])

        ax.set_ylabel('Number of DOFs')
        ax.set_xlabel(r'Local basis size $n$')
        ax.set_xticks(num_modes)
        ax.set_xticklabels(example.rom_validation.num_modes)
        ax.legend()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('hapod', type=str)
    parser.add_argument('hrrf', type=str)
    parser.add_argument('outfile', type=str)
    args = parser.parse_args(sys.argv[1:])
    main(args)
