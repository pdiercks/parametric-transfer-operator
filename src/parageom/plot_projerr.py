import numpy as np
from multi.postprocessing import read_bam_colors
from multi.plotting_context import PlottingContext


def postproc(nreal: int, method: str, distr: str, config: str, eps: float):
    from .tasks import example

    instream = example.projerr(nreal, method, distr, config)
    data = np.load(instream)
    eps2 = eps ** 2

    # find number of modes needed to achieve l2err < eps^2
    l2err = data['l2err_h1_semi']
    ind = np.argmax(l2err<eps2)

    # get corresponding nodal relative error
    relerr = data['rerr_max'][:ind+1]
    print(f"""Summary
    Need {ind} Modes to bound l2-mean error by {eps2}, which correponds to
    Need {ind} Modes to achieve relative error of {relerr[-1]}.
    """)
    breakpoint()


def main(cli):
    from .tasks import example

    distr = example.distributions[0]
    bamcd = read_bam_colors()
    blue = bamcd["blue"][0]
    red = bamcd["red"][0]

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for method in example.methods:
            infile = example.projerr(cli.nreal, method, distr, cli.config)
            data = np.load(infile)
            for key in ["l2err_h1_semi"]:
                e = data[key]
                if method == "heuristic":
                    label = "HRRF"
                    color = blue
                elif method == "hapod":
                    label = "RRF+POD"
                    color = red
                    eps = np.ones_like(e) * example.epsilon_star_projerr ** 2
                    ax.semilogy( # type: ignore
                        np.arange(e.size), eps, "k-", label=r"$(\varepsilon^{\ast})^2$"
                    )
                ax.semilogy(np.arange(e.size), e, color=color, label=label)  # type: ignore
        ax.legend(loc="best")  # type: ignore
        ax.set_xlabel("Number of basis functions")  # type: ignore
        ax.set_ylabel(r"$\ell^2$-mean projection error")  # type: ignore


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument(
        "config", type=str, help="The configuration of the oversampling problem."
    )
    parser.add_argument("outfile", type=str, help="Write plot to path (pdf).")
    args = parser.parse_args(sys.argv[1:])
    main(args)
