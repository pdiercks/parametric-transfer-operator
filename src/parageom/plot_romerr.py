import numpy as np
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(cli):
    from parageom.tasks import example

    bamcd = read_bam_colors()
    blue = bamcd['blue'][0]
    red = bamcd['red'][0]
    colors = {'hapod': red, 'hrrf': blue}
    norms = {'u': {0: 'energy', 1: 'max'}, 's': {0: 'euclidean', 1: 'max'}}
    norms = norms[cli.field]
    labels = {'hapod_max': 'RRFPOD', 'hrrf_max': 'HRRF'}
    ylabels = {
        'u': {0: r'$\mathcal{E}^{\mathcal{V}}_{u}$', 1: r'$\mathcal{E}_{u}^{\max}$'},
        's': {0: r'$\mathcal{E}^2_{\sigma}$', 1: r'$\mathcal{E}_{\sigma}^{\max}$'},
    }
    ylabels = ylabels[cli.field]

    def prepend_one(array):
        arr = np.hstack([np.ones(1), array])
        return arr

    def prepend_zero(array):
        arr = np.hstack([np.zeros(1), array])
        return arr

    data = {}
    for method in example.methods:
        data[method] = np.load(example.mean_rom_error(method, cli.field, ei=cli.ei))

    number_of_modes = prepend_zero(list(example.rom_validation.num_modes))

    args = [__file__, cli.outfile]
    styles = [example.plotting_styles['thesis'].as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots(1, 2, sharey=True)

        for k, norm in norms.items():
            for method in example.methods:
                mean = prepend_one(data[method][f'mean_{norm}_max'])
                std = prepend_zero(data[method][f'std_{norm}_max'])
                ccc = colors[method]
                label = labels.get('_'.join([method, norm]), '')
                ax[k].semilogy(number_of_modes, mean, color=ccc, linestyle='dashed', marker='.', label=label)
                ax[k].fill_between(number_of_modes, mean - std, mean + std, alpha=0.2, color=ccc)
                ax[k].set_xticks(number_of_modes)
                ax[k].set_xlabel(r'Local basis size $n$')
                ax[k].set_ylabel(ylabels[k])
        ax[-1].legend(loc='best')


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'field',
        type=str,
        help='The field for which error is plotted.',
        choices=('u', 's'),
        default='u',
    )
    parser.add_argument('outfile', type=str, help='Write plot to path (pdf).')
    parser.add_argument('--ei', action='store_true', help='Plot data of validation of ROM with EI.')
    args = parser.parse_args(sys.argv[1:])
    main(args)
