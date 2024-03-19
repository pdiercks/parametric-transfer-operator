from pymor.core.pickle import load
from multi.plotting_context import PlottingContext
from multi.postprocessing import read_bam_colors


def main(args):
    from .tasks import beam
    bamcd = read_bam_colors()

    with open(args.input, "rb") as fh:
        data = load(fh)

    jvalues = data['evaluations']
    prices = data['prices']
    compliance = data['compliance']
    evals = data['iterations']

    blue = bamcd["blue"][0]
    red = bamcd["red"][0]
    black = bamcd["black"][0]

    with PlottingContext([__file__, args.output], [beam.plotting_style.as_posix()]) as fig:
        ax = fig.subplots()
        ax.plot(evals, prices, color=blue, label="cost")
        ax.plot(evals, compliance, color=red, label="compliance")
        ax.plot(evals, jvalues, color=black, label="obj. $J$")

        ax.set_ylabel("Evaluations")
        ax.set_xlabel("Number of evaluations")
        ax.legend(loc="best")


if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(
            description="Plot optimization data."
            )
    parser.add_argument("input", type=str, help="Filepath to minimization data")
    parser.add_argument("output", type=str, help="Filepath to write PDF")
    args = parser.parse_args(sys.argv[1:])
    main(args)
