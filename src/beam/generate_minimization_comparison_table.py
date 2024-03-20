import csv
from pymor.core.pickle import load


def main(args):
    from .tasks import beam

    with beam.fom_minimization_data.open("rb") as instream:
        fom_data = load(instream)

    with beam.rom_minimization_data("normal", "heuristic").open("rb") as instream:
        rom_data = load(instream)

    table = []
    top_row = ["", "", "Beam example"]
    table.append(top_row)

    # true output
    table.append(["True output", r"$J(\mu^{\ast})$", fom_data["J(mu_min)"]])

    # reduced output
    table.append(["Reduced output", r"$J_N(\mu_N^{\ast})$", rom_data["J_N(mu_N_min)"]])

    # true output at reduced optimal solution
    table.append([r"True output at $\mu_N^{\ast}$", r"$J(\mu_N^{\ast})$", rom_data["J(mu_N_min)"]])

    # error in optimal solution μ
    table.append(["Abs. error in optimal solution", r"$\norm{\mu_N^{\ast} - \mu^{\ast}}$", rom_data["abs_err_mu"]])

    # error in output
    table.append(["Abs. error in output", r"$\vert J_N(\mu_N^{\ast}) - J(\mu^{\ast}) \vert$", rom_data["abs_err_J"]])

    # suboptimality
    table.append(["Suboptimality", r"$\frac{J(\mu_N^{\ast}) - J(\mu^{\ast})}{J(\mu^{\ast})}$", rom_data["suboptimality"]])

    with beam.minimization_comparison_table.open("w") as fh:
        header = "# terminology,formula,value\n"
        fh.write(header)
        csv_writer = csv.writer(fh, delimiter=",")
        for row in table:
            csv_writer.writerow(row)


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
