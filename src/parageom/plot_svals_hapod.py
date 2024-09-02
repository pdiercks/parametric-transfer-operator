import numpy as np
from multi.plotting_context import PlottingContext


def main(cli):
    from parageom.tasks import example

    pod = {
    0     :    "58/420",
    1     :    "80/849",
    2     :    "95/1690", 
    3     :    "157/3098",
    4     :    "155/3089",
    5     :    "157/3089",
    6     :    "156/3089",
    7     :    "157/3090",
    8     :    "78/1513",
    9     :    "74/763",
    10    :    "56/383",
    }

    args = [__file__, cli.outfile]
    styles = [example.plotting_style.as_posix()]
    with PlottingContext(args, styles) as fig:
        ax = fig.subplots()
        for k in range(11):
            sigma = np.load(example.hapod_singular_values(cli.nreal, k))
            modes = np.arange(sigma.size)
            label = rf"$k={k+1}$, {pod[k]}"
            ax.semilogy(modes, sigma / sigma[0], linestyle="dashed", label=label)
        ax.legend(loc="center left", bbox_to_anchor=(1.1, 0.5))  # type: ignore
        ax.set_xlabel(r"$n$")  # type: ignore
        ax.set_ylabel("Singular value")  # type: ignore


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser("Plot Singular values of each Oversampling Problem.")
    parser.add_argument("nreal", type=int, help="The n-th realization.")
    parser.add_argument("outfile", type=str, help="Write plot to path (pdf).")
    args = parser.parse_args(sys.argv[1:])
    main(args)
