import csv
from pymor.core.pickle import load


def main(args):
    from .tasks import beam

    with beam.fom_minimization_data.open("rb") as instream:
        fom_data = load(instream)

    with beam.rom_minimization_data("normal", "heuristic").open("rb") as instream:
        rom_data = load(instream)

    table = []
    top_row = ["Model", "Iterations", "Evaluations", "Time", "Output J"]
    table.append(top_row)

    fom_row = ["FOM", fom_data["num_iter"], fom_data["num_evals"], fom_data["time"], fom_data["J(mu_min)"]]
    table.append(fom_row)
    rom_row = ["ROM", rom_data["num_iter"], rom_data["num_evals"], rom_data["time"], rom_data["J_N(mu_N_min)"]]
    table.append(rom_row)

    with beam.minimization_data_table.open("w") as fh:

        header = "#\n" 
        fh.write(header)
        csv_writer = csv.writer(fh, delimiter=",")
        for row in table:
            csv_writer.writerow(row)


if __name__ == "__main__":
    import sys, argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
