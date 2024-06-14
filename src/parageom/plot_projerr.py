import numpy as np
from multi.plotting_context import PlottingContext


def main(cli):
    from .tasks import example
    
    distr = example.distributions[0]

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for method in example.methods:
            infile = example.projerr(cli.nreal, method, distr, cli.config)
            data = np.load(infile)
            err = data[:, 0] # relative error
            ax.semilogy(np.arange(err.size), err, label=method)
        ax.legend(loc="best")
        ax.set_xlabel("Number of basis functions")
        ax.set_ylabel("Relative projection error")
    


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("config", type=str, help="The configuration of the oversampling problem.")
    parser.add_argument("outfile", type=str, help="Write plot to path (pdf).")
    args = parser.parse_args(sys.argv[1:])
    main(args)
